# LSP-DETR: Efficient and Scalable Nuclei Segmentation in Whole Slide Images

Matěj Pekár, Vít Musil, Rudolf Nenutil, Petr Holub, Tomáš Brázdil

[[Paper](https://www.sciencedirect.com/science/article/pii/S0169260726003639)][[HF Link 🤗](https://huggingface.co/RationAI/LSP-DETR)]

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

## Citing LSP-DETR

```BibTeX
@article{pekar2026lspdetr,
  title = {LSP-DETR: Efficient and scalable nuclei segmentation in whole-slide images},
  author = {Matěj Pekár and Vít Musil and Rudolf Nenutil and Petr Holub and Tomáš Brázdil},
  journal = {Computer Methods and Programs in Biomedicine},
  volume = {287},
  pages = {109614},
  year = {2026},
  issn = {0169-2607},
  doi = {https://doi.org/10.1016/j.cmpb.2026.109614},
  url = {https://www.sciencedirect.com/science/article/pii/S0169260726003639},
}
```
