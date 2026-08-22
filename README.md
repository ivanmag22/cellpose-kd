# Cellpose-KD

A lightweight, distilled version of [Cellpose](https://github.com/MouseLand/cellpose) for fast nuclei segmentation in 3D fluorescence microscopy. The aim of this repo is to develop a small Cellpose model that runs on CPU, without sacrificing accuracy.

## Brief introduction
[Cellpose](https://github.com/MouseLand/cellpose) is a state-of-the-art segmentation model for image segmentation of cellular parts.
However, there are two relevant problems especially on 3D volumes:
- the 3D instance segmentation is computationally expensive
- the lack of manually annotated dataset

**Cellpose-KD** is a lightweight model specialized in segmentation of fluorescently labeled nuclei. Its architecture was re-designed to be more efficient and model weights were obtained through **knowledge distillation** from a pre-trained Cellpose teacher, trained on an *unlabeled* dataset of organoid images captured with confocal microscopy (20× and 40× magnification).

We also build on the design of [FastCellpose](https://github.com/YouDengdeng/FastCellpose), a lightweight variant of the baseline: it reduces the number of channels, of convolutions, and it has the transposed convolution layers instead of the upsampling layers.

*Our contribution* was to
- re-design the architecture by applying
    - dilation in encoding convolution layers
    - depthwise separable convolutions in encoding and decoding paths
- use knowledge distillation technique to transfer baseline expertise to our small models.

## Results

Cellpose-KD is **56× smaller** than baseline Cellpose (in terms of parameters and MACs) and **8.4× faster on CPU** for 3D segmentation.

### 3D inference benchmark

Here, we show how much fast is our model wrt to the other ones on 3D volumes (200-slice, 1024×1024 stack).

| Model | Parameters (K) | Inference time (min) |
|---|---|---|
| Cellpose (baseline) | 6,600 | 82.67 |
| Distilled FastCellpose-KD | 564 | 20.11 |
| **Cellpose-KD** | **115** | **9.79** |

### 2D segmentation benchmark

This model matches baseline accuracy and generalizes to external 2D benchmarks (DAPI, BitDepth).

Here, we present the results on 2D images.

| Model | Parameters (K) | Inference time (s) | F1 - ours (%) | F1 - DAPI (%) | F1 - BitDepth (%) |
|---|---|---|---|---|---|
| Cellpose (baseline) | 6,600 | 14.12 | 92.10 | 93.07 | 83.81 |
| Distilled FastCellpose-KD | 564 | 4.47 | 93.01 | 81.27 | 87.38 |
| **Cellpose-KD** | **115** | **3.30** | 92.07 | 81.60 | 87.17 |

## Requirements

```bash
pip install -r requirements.txt
```

## Repository structure

```
cellpose-kd/
├── cellpose/
│   ├── cellpose/
│   │   ├── models.py               # model definition + mask post-processing
│   │   ├── resnet_torch.py         # original Cellpose U-Net architecture
│   │   ├── dilated_resnet.py       # encoder compression (fewer layers + dilation)
│   │   ├── upsample_resnet.py      # decoder compression (+ transposed convs)
│   │   ├── downup_resnet.py        # combined encoder + decoder compression
│   │   ├── resnet_torchv2.py       # depthwise separable convolutions
│   │   ├── downup_resnetv2.py      # Cellpose-KD: combined compression + depthwise separable convs
│   │   ├── fastcp_resnet_torch.py  # FastCellpose architecture
│   │   └── __main__.py             # GUI entry point
│   ├── cellpose_metrics.py         # parameter / MACs counter
│   ├── test_cellpose.py            # inference script
│   ├── train_kd_batch.py           # teacher-student training (Cellpose-KD)
│   ├── train_kd_batch_fastcp.py    # teacher-student training (FastCellpose-KD)
│   └── utils.py                    # polynomial LR scheduler, etc.
├── daccuracy/                       # segmentation accuracy evaluation (external tool)
├── utils/
│   └── label_correction.py         # relabels masks so every object has a unique ID
└── models/
    ├── 40x/
    │   ├── cp-kd/       # Cellpose-KD checkpoints
    │   └── fastcp-kd/   # FastCellpose-KD checkpoints
    └── 20x/
        ├── cp-kd/
        └── fastcp-kd/
```

## Usage

### Run inference
Commands to launch the models (pre-trained model for 40× magnification)

**Cellpose-KD**
```bash
python cellpose/test_cellpose.py \
    --input "$file" --diameter 25 --dilation --upsample_compr \
    --nbase 16 32 64 128 --depthwise \
    --model_path models/40x/cp-kd/Best_model_epoch_20_lr1.00e-03_d25.0.pth \
    --output "$output_dir"
```

**FastCellpose-KD**
```bash
python cellpose/test_cellpose.py \
    --input "$file" --diameter 25 --fastcp --nbase 16 32 64 128 \
    --model_path models/40x/fastcp-kd/Best_model_epoch_55_lr1.00e-03_d25.0.pth \
    --output "$output_dir"
```

`test_cellpose.py` accepts 2D or 3D input, reports inference time, and saves the predicted mask (use `--no_mask` to skip mask generation).

### Evaluate against ground truth ([DAccuracy](https://src.koda.cnrs.fr/eric.debreuve/daccuracy))

```bash
# CSV ground truth
python daccuracy/package/daccuracy/cli/main.py --gt "gt.csv" --rGcF --dn "dn.tif" -s

# Image ground truth, with relabeling
python daccuracy/package/daccuracy/cli/main.py \
    --gt "gt.tif" --relabel-gt seq --dn "dn.tif" --relabel-dn seq -s
```

> DAccuracy requires every detected object to have a unique label. If your Cellpose output contains repeated labels, fix it first with `utils/label_correction.py`, which produces a `_lab_corrected` output.

### Train a model (teacher-student distillation)

Use `train_kd_batch.py` (Cellpose-KD) or `train_kd_batch_fastcp.py` (FastCellpose-KD). Key ideas:

- **Teacher**: generalist Cellpose (`model_type="nuclei"`), or any pre-trained model compatible with that model type
- **Student**: any compact variant (dilated, upsample-compressed, depthwise separable, etc.), with architecture controlled via `--nbase`
- Teacher and student use the **same diameter** for a fair comparison
- Loss is computed between teacher and student outputs directly (not against a ground-truth mask), using a modified gradient-flow loss: the gradient-flow weighting term `lambda` is set to `1` rather than the original `5`, since it compares **two model outputs** (teacher vs. student) instead of teacher's normalized output and student's output (see [Cellpose issue #1120](https://github.com/MouseLand/cellpose/issues/1120) for the reasoning)
- Validation is measured via IoU on cell probability maps (not full masks, since masks are numpy arrays and harder to handle on GPU)
- Supports resuming interrupted training, configurable validation frequency, and checkpointing every N epochs
- Supports freezing layers selectively (e.g. only transposed-conv layers, only encoder, only decoder) and loading partial weights (encoder-only, decoder-only)
- Style branches can optionally be removed
- For 2D data, pass `--patches` (recommended) or `--slices` (more prone to out-of-memory); 3D training from raw volumes is not supported by this script
- Optional data augmentation to expand the effective dataset size

## Pre-trained models

| Variant | Magnification | Path |
|---|---|---|
| Cellpose-KD | 40× | `models/40x/cp-kd/Best_model_epoch_20_lr1.00e-03_d25.0.pth` |
| Cellpose-KD | 20× | `models/20x/cp-kd/Best_model_epoch_210_lr1.00e-03_d25.0.pth` |
| FastCellpose-KD | 40× | `models/40x/fastcp-kd/Best_model_epoch_55_lr1.00e-03_d25.0.pth` |
| FastCellpose-KD | 20× | `models/20x/fastcp-kd/Best_model_epoch_240_lr1.00e-03_d25.0.pth` |

## Measuring model complexity

`cellpose/cellpose_metrics.py` reports parameter count and MACs (Multiply-Accumulate Computations) for a given input shape (default `1×2×1024×1024`, but `z`, `x`, `y` are configurable). Supports Cellpose, dilated (encoder-only compression), transposed-conv (decoder-only compression), fully compressed, depthwise separable, and pruned variants.

## GUI

```bash
python cellpose/cellpose/__main__.py
```