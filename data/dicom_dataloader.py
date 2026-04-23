import os
import pandas as pd
import torch
import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset

RADIOMICS_UID='uid'

def loadImage(filename):
    reader = sitk.ImageSeriesReader()

    dicom_names = reader.GetGDCMSeriesFileNames(filename)
    reader.SetFileNames(dicom_names)

    image = reader.Execute()

    return image

def loadMask(filename):
    reader = sitk.ImageSeriesReader()

    series_ids = reader.GetGDCMSeriesIDs(filename)
    # Each mask will only have 1 series
    idx = 0
    series_filenames = reader.GetGDCMSeriesFileNames(filename, series_ids[idx]) 
    reader.SetFileNames(series_filenames)

    mask = reader.Execute()

    return mask

class ImageDataset(Dataset):
    def __init__(self, patient_directory, patient_key=None):
        self.patient_directory = patient_directory
        self.patients = os.listdir(patient_directory)
        self.patients = [x for x in self.patients if not x.startswith('.')] #Ignores hidden files in the patient_dir
        self.image_directory = 'image'
        self.mask_directory = 'mask'
        self.patient_key = pd.read_csv(patient_key) if patient_key is not None else None
        self.multimodal_identifier = 'image'
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.patients)

class ImageClassificationDataset(ImageDataset):

    def __init__(self, patient_directory, clinical_data, patient_key=None, transforms=None):
        super().__init__(patient_directory, patient_key)
        self.classification_dataset = PreopClassificationDataset(clinical_data)
        self.transforms = transforms

    def __getitem__(self, index):
        # load image and dicomRT mask
        patient_dir = self.patients[index]
        dicom_directory = os.listdir(os.path.join(self.patient_directory, patient_dir, self.image_directory))[0]
        dicom_directory = os.path.join(self.patient_directory, patient_dir, self.image_directory, dicom_directory)
        image = loadImage(dicom_directory)

        mask_dicom_directory = os.listdir(os.path.join(self.patient_directory, patient_dir, self.mask_directory))[0]
        mask_dicom_directory = os.path.join(self.patient_directory, patient_dir, self.mask_directory, mask_dicom_directory)
        mask = loadMask(mask_dicom_directory)

        # Sometimes a dicomRT mask will have an extra dimension size 1... remove it if its there
        if mask.GetDimension() == 4 and mask.GetSize()[3] == 1:
            mask = mask[..., 0]

        # Resample mask to the image space if necessary
        mask = sitk.Resample(mask, image)

        # Convert to numpy arrays
        mask = sitk.GetArrayFromImage(mask)
        image = sitk.GetArrayFromImage(image)

        # Resampling the mask can cause some non-binary values since it interpolates voxel values, rebinarize mask here
        mask_bools = mask > 128


        ## Region of interest extraction - Apply mask and remove all slices entirely 0 valued

        # Apply mask
        masked_image = image * mask_bools

        # Removes all slices in all planes that are entirely 0 valued
        masked_image = masked_image[:, :, ~np.all(masked_image == 0, axis=(0,1))]
        masked_image = masked_image[~np.all(masked_image == 0, axis=(1,2)), :, :]
        masked_image = masked_image[:, ~np.all(masked_image == 0, axis=(0,2)), :]

        masked_image = np.array(masked_image, dtype=np.float32)

        # Get the label from corresponding clinical dataset
        uid = anonIDToRadiomicsUID(patient_dir, self.patient_key)
        uid = np.float64(uid)

        _, target = self.classification_dataset.getDataByUID(uid)
        
        # Apply any transforms, we add the batch dimension ahead of time because monai transforms expect batch data
        # In the case no transforms are being applied, we still add the extra batch dimension for consistency.
        if self.transforms is not None:
            image = self.transforms(masked_image[None, ...])
            return image, target
        return image[None, ...], target
    
