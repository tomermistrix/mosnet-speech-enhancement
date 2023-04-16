# Differentiable Mean Opinion Score Regularization for Perceptual Speech Enhancement

PyTorch implementation of our paper:

[Differentiable Mean Opinion Score Regularization for Perceptual Speech Enhancement][paper].

## Usage

### Installation in Google Colab

First, clone the repository in Google Colab notebook:

```bash
!git clone https://github.com/tomermistrix/mosnet-speech-enhancement.git
%cd mosnet-speech-enhancement
```

### MOS Estimation

Initialize the MOSNet model, load the pretrained weights and estimate MOS of a speech audio sample sampled at 16kHz:

```python
import numpy as np
import torch
import soundfile as sf
from mosnet import MOSNet

# Set device:
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set path of trained weights:
weights_path = "mosnet16_torch.pt"

# Initialize model and load weights:
mos_model = MOSNet(device=device)
mos_model.load_state_dict(torch.load(weights_path))
mos_model.eval()

# Load speech audio sample:
speech_path = "sample.wav"
y, _ = sf.read(speech_path)

# Feed speech audio sample to model and estimate MOS:
with torch.no_grad():
    y_in = torch.from_numpy(y).unsqueeze(0).to(device)
    mos_average, mos_per_frame = mos_model(y_in)

```

### Regularization for Speech Enhancement

Initialize a criterion that can be employed into any deep-learning-based training framework:

```python
import torch
from mosnet import MOSNetLoss

# Set device:
device = "cuda" if torch.cuda.is_available() else "cpu"

# Set path of trained weights:
weights_path = "mosnet16_torch.pt"

# Initialize criterion:
criterion = MOSNetLoss(weights_path, device=device)
```

## Perceptual Speech Enhancement - Audio Samples

Coming soon

## Citation

Please consider citing this work if you find it helpful in your research:

```
@article{rosenbaum2023diffmos,
  title = {Differentiable Mean Opinion Score Regularization for Perceptual Speech Enhancement},
  journal = {Pattern Recognition Letters},
  volume = {166},
  pages = {159-163},
  year = {2023},
  issn = {0167-8655},
  doi = {https://doi.org/10.1016/j.patrec.2023.01.011},
  url = {https://www.sciencedirect.com/science/article/pii/S016786552300017X},
  author = {Tomer Rosenbaum and Israel Cohen and Emil Winebrand and Ofri Gabso}
}
```


[paper]: https://www.sciencedirect.com/science/article/abs/pii/S016786552300017X
