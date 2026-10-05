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

## Code provenance and acknowledgement

SpaGate introduces an upstream feature-enhancement framework that combines
a PCA-based transcriptional representation with a spatially regularized
autoencoder representation through a parameter-free variance-guided fusion
mechanism.

The downstream dual-graph contrastive learning backbone, including related
graph construction, graph encoding, contrastive learning, and training
utilities, is adapted from the open-source spCLUE implementation:

Wang, X., Li, W. V., and Li, H. (2025).
spCLUE: a contrastive learning approach to unified spatial transcriptomics
analysis across single-slice and multi-slice data.
Genome Biology, 26, 177.

Original repository:
https://github.com/EnchantedJoy/spCLUE

spCLUE is distributed under the MIT License. The original copyright notice
and license are retained in `LICENSES/spCLUE_LICENSE.txt`.
