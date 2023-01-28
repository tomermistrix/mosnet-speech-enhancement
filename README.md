# Differentiable Mean Opinion Score Regularization for Perceptual Speech Enhancement

PyTorch implementation of our paper: [Differentiable Mean Opinion Score Regularization for Perceptual Speech Enhancement][paper].

## Usage

### MOS Estimation

Initialize the MOSNet model, load the pretrained weights and estimate MOS of a speech audio sample:

```python
import numpy as np
import torch
import librosa
from mosnet import MOSNet

# Set some parameters:
device = "cuda" if torch.cuda.is_available() else "cpu"
sr = 16000

# Set path of trained weights:
weights_path = "mosnet16_torch.pt"

# Initialize model and load weights:
mos_model = MOSNet(device=device)
mos_model.load_state_dict(torch.load(weights_path))
mos_model.eval()

# Load audio sample and estimate MOS:
speech_path = "sample.wav"
y, _ = librosa.load(speech_path, sr=sr)

# Feed to model and estimate MOS:
with torch.no_grad():
    y_in = torch.from_numpy(y).unsqueeze(0).to(device)
    avg_mos, mos_frames = mos_model(y_in)

```

### Regularization for Speech Enhancement

Initialize a criterion that can be employed into any deep-learning-based training framework:

```python
import torch
from mosnet import MOSNetLoss

# Set some parameters:
device = "cuda" if torch.cuda.is_available() else "cpu"
sr = 16000

# Set path of trained weights:
weights_path = "mosnet16_torch.pt"

# Initialize criterion:
criterion = MOSNetLoss(weights_path, device=device)
```

## Audio Samples

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


[paper]: https://authors.elsevier.com/a/1gUxL_3qHiVA7n
