# HMPL Project - Wearing of Face Masks Detection Using CNN

## 1. How to execute the code
Before run, make sure having the following dependencies:
- Pytorch
- Pillow

If run on CPU, use:
`python3 project.py`

If run on GPU, use:
`python3 --use_cuda 1 project.py`

## 2. Code structure
`class MaskDataSet(Dataset):
...`
### Custom dataset class extended Pytorch Dataset, use to store and process pictures and apply transforms from inputs.

## 3. Performance results
