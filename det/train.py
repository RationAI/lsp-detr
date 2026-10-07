"""LSP-DETR detection training entry point.

Same flow as Dome's ``train.py`` (YAMLConfig -> DetSolver.fit) with two additions:
  * ``import lsp_det`` bootstraps the Dome repo onto ``sys.path`` and registers the LSP components;
  * the optimizer param groups are audited (printed + asserted) before ``fit``.

Usage (see det/README.md):
  torchrun --standalone --nproc_per_node=8 det/train.py -c det/configs/LSP-T-combined.yml \
      --output-dir <dir> --seed 0 -u epoches=30 train_dataloader.total_batch_size=8 val_dataloader.total_batch_size=8
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # det/ -> `import lsp_det`

import lsp_det  # noqa: E402  (bootstraps Dome + registers LSP components)
from src.core import YAMLConfig, yaml_utils  # noqa: E402
from src.misc import dist_utils  # noqa: E402
from src.solver import TASKS  # noqa: E402


def _apply_tf32_policy(mode: str) -> None:
    """This machine forces TF32 for cuBLAS by env (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1), i.e. the reference Dome
    runs trained with TF32 matmuls although no code sets it. 'keep' = machine default (Dome-actual parity),
    'off' = pure FP32 (strategy doc §7 wording), 'on' = force TF32."""
    import torch
    if mode == "off":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
    elif mode == "on":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"[tf32] mode={mode} matmul.allow_tf32={torch.backends.cuda.matmul.allow_tf32} cudnn.allow_tf32={torch.backends.cudnn.allow_tf32} "
          f"float32_matmul_precision={torch.get_float32_matmul_precision()} TORCH_ALLOW_TF32_CUBLAS_OVERRIDE={os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')}")


def main(args) -> None:
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)
    _apply_tf32_policy(args.tf32)

    assert not all([args.tuning, args.resume]), "Only support from_scrach or resume or tuning at one time"

    update_dict = yaml_utils.parse_cli(args.update)
    update_dict.update({k: v for k, v in args.__dict__.items()
                        if k not in ["update", "skip_optim_audit", "tf32", "expect_backbone_lr"] and v is not None})

    if lsp_det.CRITERION_IMPORT_ERROR is not None:
        raise ImportError("LSPCriterion is unavailable, so training cannot run: "
                          f"{lsp_det.CRITERION_IMPORT_ERROR} (check $DOME_ROOT)")

    cfg = YAMLConfig(args.config, **update_dict)
    print("cfg: ", cfg.__dict__)

    if not args.skip_optim_audit and "optimizer" in cfg.yaml_cfg:
        # builds cfg.model (cached; the solver reuses the same object) and checks the regex groups
        lsp_det.optim_audit.audit_param_groups(cfg.model, cfg.yaml_cfg["optimizer"],
                                               expect={"backbone_lr": args.expect_backbone_lr, "min_groups": 3}, verbose=True)
        rep = getattr(cfg.model, "pretrained_report", None)
        if rep is not None:
            print(f"[hf5class] init: arm={rep['arm']} loaded_tensors={rep['loaded_tensors']} loaded_params={rep['loaded_params']:,} "
                  f"coverage={rep['coverage']*100:.2f}% new_params={rep['new_params']:,}")
        # persist init metadata next to the checkpoints (arm / point-output init / wh prior / tf32) - §9 checklist
        if dist_utils.is_main_process() and cfg.output_dir:
            import json, torch
            os.makedirs(cfg.output_dir, exist_ok=True)
            m = cfg.model
            meta = {"model": cfg.yaml_cfg.get("model"), "center_mode": getattr(m, "center_mode", None),
                    "center_span_cells": getattr(getattr(m, "decoder", None), "center_span", None),
                    "wh_prior_px": cfg.yaml_cfg.get("LSPDetrDetection", {}).get("wh_prior_px"),
                    "point_output_init": ("loaded from hf-5class" if getattr(m, "center_mode", None) == "strict-local" else "zero re-init (movable-reference)"),
                    "wh_output_init": "zero", "class_head_init": "new, bias prior 0.01",
                    "tf32": {"mode": args.tf32, "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "float32_matmul_precision": torch.get_float32_matmul_precision()},
                    "frozen_backbone_tensors": getattr(m, "frozen_backbone_tensors", None),
                    "pretrained_report": rep}
            with open(os.path.join(cfg.output_dir, "init_report.json"), "w") as f:
                json.dump(meta, f, indent=2)

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)
    if args.test_only:
        solver.val()
    else:
        solver.fit()
    dist_utils.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, help="resume from checkpoint")
    parser.add_argument("-t", "--tuning", type=str, help="tuning from checkpoint")
    parser.add_argument("-d", "--device", type=str, help="device")
    parser.add_argument("--seed", type=int, help="exp reproducibility")
    parser.add_argument("--use-amp", action="store_true", help="auto mixed precision training")
    parser.add_argument("--output-dir", type=str, help="output directoy")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument("--test-only", action="store_true", default=False)
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")
    parser.add_argument("--local-rank", type=int, help="local rank id")
    parser.add_argument("--skip-optim-audit", action="store_true", default=False)
    parser.add_argument("--expect-backbone-lr", type=float, default=1.25e-5,
                        help="LR the optimizer audit expects on the backbone groups (LSP-T-combined recipe: 1.25e-5)")
    parser.add_argument("--tf32", type=str, default="keep", choices=["keep", "off", "on"], help="TF32 policy (see _apply_tf32_policy)")
    args = parser.parse_args()
    main(args)
