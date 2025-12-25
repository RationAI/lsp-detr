# LSP-DETR: Efficient and Scalable Nuclei Segmentation in Whole Slide Images

LSP-DETR (Local Star Polygon DEtection TRansformer) is a lightweight, efficient, and end-to-end deep learning model for nuclei instance segmentation in histopathological images. It combines a DETR-based transformer decoder with star-convex polygon shape descriptors to enable accurate and fast segmentation without complex post-processing.


## Installation

To install the necessary dependencies, follow these steps:

```bash
git clone https://github.com/RationAI/lsp-detr.git
cd lsp-detr
uv sync
```

## Training on PanNuke

You need at least 10Gb of GPU memory to train the model.

```bash
uv run -m lsp_detr +experiment=PanNuke +data.train_fold=1 +data.val_fold=2 +data.test_fold=3
```
