# DGU-code

We have currently uploaded the code based on AAPM dataset.
Before running the code, in addition to the commonly used deep learning libraries, you also need to install the following dependencies:

* [MONAI](https://github.com/Project-MONAI/MONAI)
* [MONAI Generative Models](https://github.com/Project-MONAI/GenerativeModels)
* [torch-radon](https://github.com/matteo-ronchetti/torch-radon)

> **Note:** When installing `torch-radon`, you may need to apply a patch provided by [helix2fan](https://github.com/faebstn96/helix2fan) in order to install it properly.

To train DGU, simply run:

```bash
python train_AAPM.py
```
