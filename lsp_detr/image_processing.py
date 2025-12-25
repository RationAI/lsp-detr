import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from torch import Tensor
from transformers import BaseImageProcessorFast


class LSPDetrImageProcessor(BaseImageProcessorFast):
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)
    do_rescale = True
    do_normalize = True

    def post_process(self, outputs: dict[str, Tensor]) -> list[dict[str, Tensor]]:
        """Converts the raw output into polygons.

        Returns:
            A list of dictionaries, each containing:
                - "polygons": A tensor of shape (N, num_radial_distances, 2) representing the polygons.
                - "labels": A tensor of shape (N,) representing the labels for each polygon.
        """
        radial_distances = outputs["radial_distances"].exp()

        t = torch.linspace(
            0, 1, radial_distances.size(-1) + 1, device=radial_distances.device
        )[:-1]
        cos = torch.cos(2 * torch.pi * t)
        sin = torch.sin(2 * torch.pi * t)

        polar = radial_distances.unsqueeze(-1) * torch.stack([sin, cos], dim=-1)
        polygons = outputs["absolute_points"].unsqueeze(-2) + polar

        labels = outputs["logits"].argmax(dim=-1)
        non_no_object_indices = labels != outputs["logits"].size(-1) - 1

        return [
            {"polygons": polygons[b, indices], "labels": labels[b, indices]}
            for b, indices in enumerate(non_no_object_indices)
        ]

    def post_process_instance(
        self,
        results: list[dict[str, Tensor]],
        height: int,
        width: int,
        allow_overlap: bool = True,
    ) -> list[dict[str, Tensor]]:
        """Converts the output into actual instance segmentation predictions.

        Args:
            results: Results list obtained by `post_process`, to which "masks" results will be added.
            height: Height of the input image.
            width: Width of the input image.
            allow_overlap: If True, masks can overlap. If False, larger nuclei will cover smaller ones.
        """
        for i, result in enumerate(results):
            masks = torch.zeros(
                (len(result["polygons"]), height, width),
                dtype=torch.bool,
                device=result["polygons"].device,
            )

            for j, polygon in enumerate(result["polygons"]):
                img = Image.fromarray(masks[j].cpu().numpy())
                canvas = ImageDraw.Draw(img)
                canvas.polygon(xy=polygon.flatten().tolist(), outline=1, fill=1)
                masks[j] = torch.tensor(np.asarray(img))

            if not allow_overlap:
                # Calculate area for each mask
                areas = masks.sum(dim=(1, 2))
                # Sort indices by area (largest first)
                sorted_indices = torch.argsort(areas, descending=True)

                # Create a composite mask to track occupied pixels
                occupied = torch.zeros(
                    (height, width), dtype=torch.bool, device=masks.device
                )

                # Process masks from largest to smallest
                for idx in sorted_indices:
                    # Remove pixels that are already occupied by larger nuclei
                    masks[idx] = masks[idx] & ~occupied
                    # Update occupied pixels
                    occupied = occupied | masks[idx]

            results[i]["masks"] = masks

        return results

    def resolve_nuclei_overlaps(self, masks: Tensor) -> Tensor:
        """Resolves overlapping nuclei masks using Distance Transform.

        Splits overlaps based on which instance center is closer.

        Args:
            masks: [N, H, W] Boolean or Binary Tensor (0 or 1).

        Returns:
            flattened: [N, H, W] Boolean Tensor (Non-overlapping)
        """
        if masks.numel() == 0:
            return masks

        n, h, w = masks.shape

        masks_np = masks.detach().cpu().numpy().astype(bool)
        dist_maps = np.zeros((n, h, w), dtype=np.float32)

        for i in range(n):
            if masks_np[i].any():
                # distance_transform_edt computes Euclidean distance to background
                dist_maps[i] = distance_transform_edt(masks_np[i])

        dist_tensor = torch.from_numpy(dist_maps).to(masks.device)

        # 3. Identify the "Owner" of each pixel
        # We look at the entire stack. For every pixel (h,w), find which
        # instance 'n' has the highest distance value.

        # 'max_indices' tells us: "If this pixel belongs to anyone, it belongs to Instance X"
        _, max_indices = torch.max(dist_tensor, dim=0)  # Shape [H, W]

        # 4. Filter by original prediction
        # A pixel is only valid for Instance X if:
        #   a) Instance X has the highest distance score there (it "won" the pixel)
        #   b) Instance X actually predicted that pixel (dist > 0)

        # Create a grid to compare against max_indices
        instance_ids = torch.arange(n, device=masks.device)[:, None, None]  # [N, 1, 1]

        return (instance_ids == max_indices) & (dist_tensor > 0)
