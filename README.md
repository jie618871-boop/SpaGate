# SpaGate

Adaptive Multi-view Representation Learning for Spatial Domain Identification in Spatial Transcriptomics


## Overview

SpaGate is an adaptive multi-view representation learning framework designed for spatial domain identification (SDI) in spatial transcriptomics (ST).

SpaGate improves node representations by integrating complementary transcriptional and spatial information. It constructs a PCA-based representation and a spatially regularized autoencoder representation, followed by an adaptive variance-based gating strategy to generate enhanced feature representations for downstream graph-based learning.

The framework supports both single-slice spatial domain identification and multi-slice spatial transcriptomics integration.


## Requirements

- Python >= 3.10
- PyTorch >= 2.1.2
- CUDA 11.8 (recommended for GPU acceleration)

Install the required Python packages:

```bash
pip install -r requirements.txt