import os
import torch
import random
import pydicom
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

def zero_outside_circle(image):
    height, width = image.shape
    assert height == width, "Image must be square"
    center = (height // 2, width // 2)
    radius = height // 2
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    circular_mask = dist_from_center <= radius
    image[~circular_mask] = -1024
    return image






class CTDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir

        self.patients = [d for d in os.listdir(root_dir)]
        self.slice_paths = []
        self.patient_slices = {} 
    
        idx = 0
        for patient in self.patients:
            patient_dir = os.path.join(root_dir, patient)
            slices = sorted([f for f in os.listdir(patient_dir) if f.endswith('.dcm')],
                            key=lambda x: int((x.replace('.', '-').split('-'))[1]))
            
            patient_slice_count = len(slices)
            self.patient_slices[patient] = (idx, idx + patient_slice_count - 1)
            self.slice_paths += [os.path.join(patient_dir, s) for s in slices]
            idx += patient_slice_count


    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, idx):
        data = {}

        slice_path = self.slice_paths[idx]
        
        
        dicom_data = pydicom.dcmread(slice_path)
        image = dicom_data.pixel_array.astype(np.float32)
        rescaled_image = image * float(dicom_data.RescaleSlope) + float(dicom_data.RescaleIntercept)
        rescaled_image = zero_outside_circle(rescaled_image)
        rescaled_image = torch.from_numpy(rescaled_image).unsqueeze(0)
        rescaled_image = torch.clamp(rescaled_image, min=-1000, max=400)
        data["image"] = (rescaled_image * (1/1400) + (5/7)).float()
        data["path"] = slice_path
        return data
