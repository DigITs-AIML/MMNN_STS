import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import random
import nibabel as nib
from monai.transforms import \
    Compose, Resize, AsChannelFirst, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, EnsureChannelFirst
import sys

from data.constants import RADIOMICS_UID
from data.utils import loadImage, loadMask, anonIDToRadiomicsUID
from data.s3utils import getNiftiFilenames, loadNifti, readCSVS3, parseS3ImageFolder, loadDicom
from data.ClinicalDatasets import PreopSurvivalDataset, PreopClassificationDataset
from exceptions.exceptions import *

#suppresses sitk warnings (warns for trivial nonuniformities)
sitk.ProcessObject_SetGlobalWarningDisplay(False)


def getIndexFromValue(df, value):
        return df.index[df == value].item()

class ImageDataset(Dataset):
    def __init__(self, patient_directory, patient_key):
        self.patient_directory = patient_directory
        self.patients = os.listdir(patient_directory)
        self.patients = [x for x in self.patients if not x.startswith('.')]
        self.image_directory = 'image'
        self.mask_directory = 'mask'
        self.patient_key = pd.read_csv(patient_key)
        self.multimodal_identifier = 'image'
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.patients)

    def getDataByUID(self, uid):
        patient_ids = self.patient_key.loc[self.patient_key[RADIOMICS_UID] == uid]
        anon_id = patient_ids['Anon MRN'].item()

        index = self.patients.index(anon_id)
        return self.__getitem__(index)
    
    @property
    def uids(self):
        uids = []
        for anon_id in self.patients:
            patient_ids = self.patient_key.loc[self.patient_key['Anon MRN'] == anon_id] # Old keys use 'Anon Name'
            mrn = patient_ids[RADIOMICS_UID].item()
            uids.append(int(mrn))
        return uids

class S3ImageDataset(Dataset):
    def __init__(self, patients, patient_key, nifti=True):

        #TODO : Align expected preprocessing of anon MRNs (self.patients) BEFORE initializing parent class between s3 nifti and dicom datasets
        
        if nifti:
            # get the filename of s3 objects for each patient
            self.patients = ['-'.join(x[0].split('/')[-1].split('-')[:2]) for x in patients]

            # Remove the scan_ prefix and the file extension for the anon id
            self.patients = [x.replace('scan_', '').replace('.nii.gz', '') for x in self.patients]
        else:
            self.patients = patients
        # Read patient key
        self.patient_key = readCSVS3(patient_key)
        self.multimodal_identifier = 'image'
    
    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.patients)

    def getDataByUID(self, uid):
        patient_ids = self.patient_key.loc[self.patient_key[RADIOMICS_UID] == uid]
        anon_id = patient_ids['Anon MRN'].item()

        index = self.patients.index(anon_id)
        return self.__getitem__(index)
    
    @property
    def uids(self):
        uids = []
        for anon_id in self.patients:
            patient_ids = self.patient_key.loc[self.patient_key['Anon MRN'] == anon_id]
            try:
                mrn = patient_ids[RADIOMICS_UID].item()
            except Exception as e:
                raise InitializationError('Could not find UID for patient {} - Ensure uid is in the patient key'.format(anon_id))
            
            uids.append(int(mrn))
        return uids

class S3DicomDataset(S3ImageDataset):
    def __init__(self, patient_directory, clinical_data, patient_key, transforms = None, slices=False):
        '''
        patient_directory - path to s3 directory containing list of patients with images
            
            Expected structure of the directory is ....

                Anon MRN 1
                    'image'
                        one .dcm file for every slice in MRI
                    'mask'
                        one .dcm DicomRT file for the segmentation mask
                Anon MRN 2
                    'image'
                        one .dcm file for every slice in MRI
                    'mask'
                        one .dcm DicomRT file for the segmentation mask
                Anon MRN 3
                    .....

                ..... 
                
                for each patient
        clinical data - path to csv file containing clinical data for every patient
        patient key - path to csv file containing patient key (maps anon MRNs to MRNs)
        '''
        
        # get the anon MRNs for each patient in the dataset
        self.dcm_dict = parseS3ImageFolder(patient_directory)
        self.patients = list(self.dcm_dict.keys())
        super().__init__(self.patients, patient_key, nifti=False)
        self.classification_dataset = PreopClassificationDataset(clinical_data)
        self.transforms = transforms
        self.slices = slices
    
    def __getitem__(self, index):
        anon_mrn = self.patients[index]

        image = loadDicom(self.dcm_dict[anon_mrn]['image'])
        mask = loadDicom(self.dcm_dict[anon_mrn]['mask'])

        if mask.GetDimension() == 4 and mask.GetSize()[3] == 1:
            mask = mask[..., 0]

        # Resample mask to the image space if necessary
        mask = sitk.Resample(mask, image)

        # Convert to numpy
        mask = sitk.GetArrayFromImage(mask)
        image = sitk.GetArrayFromImage(image)

        # Resampling the mask can cause some non-binary values since it interpolates voxel values, rebinarize mask here
        mask_bools = mask > 128

        # Apply mask
        masked_image = image * mask_bools

        # Removes all slices in all planes that are entirely 0 valued
        masked_image = masked_image[:, :, ~np.all(masked_image == 0, axis=(0,1))]
        masked_image = masked_image[~np.all(masked_image == 0, axis=(1,2)), :, :]
        masked_image = masked_image[:, ~np.all(masked_image == 0, axis=(0,2)), :]

        masked_image = np.array(masked_image, dtype=np.float32)

        if self.slices:
            slice_index = int(random.random()*image.shape[0])
            image = image[slice_index, ...]

        #label
        uid = anonIDToRadiomicsUID(anon_mrn, self.patient_key)
        uid = np.float64(uid)
        mrns = self.classification_dataset.targets[RADIOMICS_UID]
        classification_index = getIndexFromValue(mrns, uid)
        _, target = self.classification_dataset.__getitem__(classification_index)

        # input_validation = 0
        # if input_validation:
        #     mask = np.array(mask_bools * 255, dtype=np.float32)
        #     rand_string = str(int(random.random() * 1000))
        #     validation_image = nib.Nifti1Image(masked_image, affine=np.eye(4))
        #     validation_mask = nib.Nifti1Image(mask, affine=np.eye(4))
        #     nib.save(validation_image, "./input_validation/image_validation_{}.nii.gz".format(rand_string))
        #     nib.save(validation_mask, "./input_validation/mask_validation_{}.nii.gz".format(rand_string))
        
        if self.transforms is not None:
            image = self.transforms(masked_image[None, ...])
            if self.slices:
                slice_index = int(random.random()*image.shape[-1])
                image = image[..., slice_index]
            return image, target
        return image[None, ...], target

    def __len__(self):
        return len(self.patients)
        
class ImageSurvivalDataset(ImageDataset):

    def __init__(self, patient_directory, clinical_data, patient_key):
        super().__init__(patient_directory, patient_key)

        #preop or postop doesn't matter - we just want the targets
        self.survival_dataset = PreopSurvivalDataset(clinical_data)
        
    
    def __getitem__(self, index):

        #image
        patient_dir = self.patients[index]

        # Only one dicom per patient so we can use index 0
        dicom_directory = os.listdir(os.path.join(self.patient_directory, patient_dir, self.image_directory))[0]
        dicom_directory = os.path.join(self.patient_directory, patient_dir, self.image_directory, dicom_directory)
        image = loadImage(dicom_directory)

        #label
        uid = anonIDToRadiomicsUID(patient_dir, self.patient_key)
        uid = np.float64(uid)
        mrns = self.survival_dataset.targets[RADIOMICS_UID]
        survival_index = getIndexFromValue(mrns, uid)
        _, target = self.survival_dataset.__getitem__(survival_index)

        return image, target

class ImageClassificationDataset(ImageDataset):

    def __init__(self, patient_directory, clinical_data, patient_key, slices=False, transforms=None):
        super().__init__(patient_directory, patient_key)
        self.classification_dataset = PreopClassificationDataset(clinical_data)
        self.slices = slices
        self.transforms = transforms

    def __getitem__(self, index):
        #image
        patient_dir = self.patients[index]
        dicom_directory = os.listdir(os.path.join(self.patient_directory, patient_dir, self.image_directory))[0]
        dicom_directory = os.path.join(self.patient_directory, patient_dir, self.image_directory, dicom_directory)
        image = loadImage(dicom_directory)

        mask_dicom_directory = os.listdir(os.path.join(self.patient_directory, patient_dir, self.mask_directory))[0]
        mask_dicom_directory = os.path.join(self.patient_directory, patient_dir, self.mask_directory, mask_dicom_directory)
        mask = loadMask(mask_dicom_directory)

        if mask.GetDimension() == 4 and mask.GetSize()[3] == 1:
            mask = mask[..., 0]

        # Resample mask to the image space if necessary
        mask = sitk.Resample(mask, image)

        # Convert to numpy
        mask = sitk.GetArrayFromImage(mask)
        image = sitk.GetArrayFromImage(image)

        # Resampling the mask can cause some non-binary values since it interpolates voxel values, rebinarize mask here
        mask_bools = mask > 128

        # Apply mask
        masked_image = image * mask_bools

        # Removes all slices that are entirely 0 valued
        masked_image = masked_image[:, :, ~np.all(masked_image == 0, axis=(0,1))]
        masked_image = masked_image[~np.all(masked_image == 0, axis=(1,2)), :, :]
        masked_image = masked_image[:, ~np.all(masked_image == 0, axis=(0,2)), :]

        masked_image = np.array(masked_image, dtype=np.float32)

        if self.slices:
            slice_index = int(random.random()*image.shape[0])
            image = image[slice_index, ...]

        #label
        uid = anonIDToRadiomicsUID(patient_dir, self.patient_key)
        uid = np.float64(uid)
        mrns = self.classification_dataset.targets[RADIOMICS_UID]
        classification_index = getIndexFromValue(mrns, uid)
        _, target = self.classification_dataset.__getitem__(classification_index)

        # input_validation = 0
        # if input_validation:
        #     mask = np.array(mask_bools * 255, dtype=np.float32)
        #     rand_string = str(int(random.random() * 1000))
        #     validation_image = nib.Nifti1Image(masked_image, affine=np.eye(4))
        #     validation_mask = nib.Nifti1Image(mask, affine=np.eye(4))
        #     nib.save(validation_image, "./input_validation/image_validation_{}.nii.gz".format(rand_string))
        #     nib.save(validation_mask, "./input_validation/mask_validation_{}.nii.gz".format(rand_string))
        
        if self.transforms is not None:
            image = self.transforms(masked_image[None, ...])
            if self.slices:
                slice_index = int(random.random()*image.shape[-1])
                image = image[..., slice_index]
            return image, target
        return image[None, ...], target


class ImageSegmentationDataset(ImageDataset):

    def __init__(self, patient_directory):
        super().__init__(patient_directory)

    def __getitem__(self, index):
        patient_dir = self.patients[index]
        image_dicom_directory = os.listdir(os.path.join(self.patient_directory, patient_dir, self.image_directory))[0]
        image_dicom_directory = os.path.join(self.patient_directory, patient_dir, self.image_directory, image_dicom_directory)

        mask_dicom_directory = os.listdir(os.path.join(self.patient_directory, patient_dir, self.mask_directory))[0]
        mask_dicom_directory = os.path.join(self.patient_directory, patient_dir, self.mask_directory, mask_dicom_directory)

        return loadImage(image_dicom_directory), loadMask(mask_dicom_directory)

class ImageDatasetByUIDs(ImageDataset):
    '''
    This dataset is initialized with a dataset and a list of uids.

    The class acts as a sub-dataset to the original that can only access data for the uids provided at initialization.
    '''
    def __init__(self, dataset, uids, seed=42, train_percent=0.8, transforms=None):
        self.dataset = dataset
        self.dataset.transforms = transforms
        self.set_uids = uids

    def __getitem__(self, index):
        return self.dataset.getDataByUID(self.set_uids[index])

    def __len__(self):
        return len(self.set_uids)

class NiftiImageDataset(ImageDataset):
    def __init__(self, patient_directory, clinical_data, patient_key, slices= False, transforms=None):
        super().__init__(patient_directory, patient_key)
        self.patient_filenames = self.patients
        self.patients = ['-'.join(x.split('-')[:2]) for x in self.patients]
        self.classification_dataset = PreopClassificationDataset(clinical_data)
        self.transforms = transforms
        self.slices = slices

    def __getitem__(self, index):
        patient_dir = self.patient_filenames[index]
        uid = self.uids[index]
        all_filenames = os.listdir(os.path.join(self.patient_directory, patient_dir))
        image = None
        mask = None

        # Load scans
        for x in all_filenames:

            # TODO: distinguish between mask and image without relying on naming conventions
            if x.startswith('scan'):
                image = nib.load(os.path.join(self.patient_directory, patient_dir, x)).get_fdata()
            else:
                mask = nib.load(os.path.join(self.patient_directory, patient_dir, x)).get_fdata()
        # Apply mask
        masked_image = image * mask

        # Removes all slices that are entirely 0 valued
        masked_image = masked_image[:, :, ~np.all(masked_image == 0, axis=(0,1))]

        # Removes all slices in all planes that are entirely 0 valued
        masked_image = masked_image[~np.all(masked_image == 0, axis=(1,2)), :, :]
        masked_image = masked_image[:, ~np.all(masked_image == 0, axis=(0,2))]

        # Get target
        uid = np.float64(uid)
        mrns = self.classification_dataset.targets[RADIOMICS_UID]
        classification_index = getIndexFromValue(mrns, uid)
        _, target = self.classification_dataset.__getitem__(classification_index)


        if self.transforms is not None:
            image = self.transforms(masked_image[None, ...])
            if self.slices:
                slice_index = int(random.random()*masked_image.shape[-1])
                image = image[..., slice_index]
            return image, target
        return masked_image[None, ...], target
    
    def __len__(self):
        return len(self.patients)
    
class S3NiftiImageDataset(S3ImageDataset):
    def __init__(self, prefix, clinical_data, patient_key, transforms=None, slices=False):
        patients = getNiftiFilenames(prefix)
        super().__init__(patients, patient_key, nifti=True)
        self.patient_filenames = patients

        # Preop or postop doesn't matter - we'll only need the target variables
        self.classification_dataset = PreopClassificationDataset(clinical_data)
        self.transforms = transforms
        self.slices = slices

    def __getitem__(self, index):
        patient_files = self.patient_filenames[index]
        uid = self.uids[index]

        image = loadNifti(patient_files[0])
        mask = loadNifti(patient_files[1])
    
        # Apply mask
        masked_image = image * mask

        # Removes all slices in all planes that are entirely 0 valued
        masked_image = masked_image[:, :, ~np.all(masked_image == 0, axis=(0,1))]
        masked_image = masked_image[~np.all(masked_image == 0, axis=(1,2)), :, :]
        masked_image = masked_image[:, ~np.all(masked_image == 0, axis=(0,2)), :]

        # Get target
        uid = np.float64(uid)
        mrns = self.classification_dataset.targets[RADIOMICS_UID]
        classification_index = getIndexFromValue(mrns, uid)
        _, target = self.classification_dataset.__getitem__(classification_index)
        
        if self.transforms is not None:
            image = self.transforms(masked_image[None, ...])
            if self.slices:
                slice_index = int(random.random()*masked_image.shape[-1])
                image = image[..., slice_index]
            return image, target
        return masked_image[None, ...], target
    
    def __len__(self):
        return len(self.patients)
    
class NiftiSurvivalDataset(ImageDataset):
    def __init__(self, patient_directory, clinical_data, patient_key, slices= False, transforms=None):
        super().__init__(patient_directory, patient_key)
        self.patient_filenames = self.patients
        self.patients = ['-'.join(x.split('-')[:2]) for x in self.patients]
        self.survival_dataset = PreopSurvivalDataset(clinical_data)
        self.transforms = transforms
        self.slices = slices

    def __getitem__(self, index):
        patient_dir = self.patient_filenames[index]
        uid = self.uids[index]
        all_filenames = os.listdir(os.path.join(self.patient_directory, patient_dir))
        image = None
        mask = None

        # Load scans
        for x in all_filenames:
            if x.startswith('scan'):
                image = nib.load(os.path.join(self.patient_directory, patient_dir, x)).get_fdata()
            else:
                mask = nib.load(os.path.join(self.patient_directory, patient_dir, x)).get_fdata()
        # Apply mask
        masked_image = image * mask

        # Removes all slices that are entirely 0 valued
        masked_image = masked_image[:, :, ~np.all(masked_image == 0, axis=(0,1))]

        # Removes all slices in all planes that are entirely 0 valued
        masked_image = masked_image[~np.all(masked_image == 0, axis=(1,2)), :, :]
        masked_image = masked_image[:, ~np.all(masked_image == 0, axis=(0,2))]
        # masked_image = masked_image[~np.all(masked_image==0, axis=(1,2)), ~np.all(masked_image==0, axis=(0,2)), ~np.all(masked_image == 0, axis=(0,1))]

        # Get target
        uid = np.float64(uid)
        mrns = self.survival_dataset.targets[RADIOMICS_UID]
        survival_index = getIndexFromValue(mrns, uid)
        _, target = self.survival_dataset.__getitem__(survival_index)
        # mrn = target.pop(RADIOMICS_UID)
        events = torch.Tensor([int(target[key][0]) for key in target.keys()])
        durations = torch.Tensor([target[key][1] for key in target.keys()])

        if self.transforms is not None:
            image = self.transforms(masked_image[None, ...])
            if self.slices:
                slice_index = int(random.random()*masked_image.shape[-1])
                image = image[..., slice_index]
            return image, events, durations
        return masked_image[None, ...], events, durations
    
    def __len__(self):
        return len(self.patients)

class S3NiftiSurvivalDataset(S3ImageDataset):
    def __init__(self, prefix, clinical_data, patient_key, transforms=None, slices=False):
        patients = getNiftiFilenames(prefix)
        super().__init__(patients, patient_key, nifti=True)
        self.patient_filenames = patients
        self.survival_dataset = PreopSurvivalDataset(clinical_data)
        self.transforms = transforms
        self.slices = slices

    def __getitem__(self, index):
        patient_files = self.patient_filenames[index]
        uid = self.uids[index]

        image = loadNifti(patient_files[0])
        mask = loadNifti(patient_files[1])
        # Apply mask
        masked_image = image * mask

        image_max = np.max(masked_image)

        first_idxs = np.all(masked_image == 0, axis=(0,1))
        second_idxs = np.all(masked_image == 0, axis=(1,2))
        third_idxs = np.all(masked_image == 0, axis=(0,2))

        masked_image[:, :, first_idxs] = image_max / 2
        masked_image[second_idxs, :, :] = image_max / 2
        masked_image[:, third_idxs, :] = image_max / 2

        # Get target
        uid = np.float64(uid)
        mrns = self.survival_dataset.targets[RADIOMICS_UID]
        survival_index = getIndexFromValue(mrns, uid)
        _, events, durations = self.survival_dataset.__getitem__(survival_index)

        if self.transforms is not None:
            image = self.transforms(masked_image[None, ...])
            if self.slices:
                slice_index = int(random.random()*masked_image.shape[-1])
                image = image[..., slice_index]
            return image, events, durations
        return masked_image[None, ...], events, durations
    
    def __len__(self):
        return len(self.patients)
    
class T1T2ImageDataset(S3ImageDataset):

    '''
    This dataset takes a S3NiftiImageDataset for t1 and t2 images

    It will load the t1 and t2 images, stack them along the channel dimension, then apply augmentations to the joint 2-channel image
    '''
    def __init__(self, t1_directory, t2_directory, clinical_data, patient_key, slices= False, transforms=None):

        # We'll initialize the parent class here using T1 data so we have access to the functions present.
        # Where needed, we'll overload member variables to account for the inclusion of t2 images (particularly, self.patients)
        # If we expand on this later, it would be useful to create a new base class that does not assume only one image modality
        # mainly for cleanliness/maintainability, this works fine for now
        super().__init__(t1_directory, patient_key)
        self.t1_dataset = None
        self.t2_dataset = None
        self.transforms = transforms

        sub_transforms = Compose([
            EnsureChannelFirst(channel_dim=0),
            Resize(spatial_size=(64,64,64))
        ])

        # It seems a little disjointed to apply separate transforms (augmentations) to the individual t1/t2 images before they are concatenated
        # We'll initialize our sub-datasets with only transforms required to concatenate them without issue, 
        # We'll apply the rest of the transforms in the __getitem__ function for this class
        # That way, both the T1 and T2 images can be concatenated without issue while still receiving the same augmentations

        self.t1_dataset = S3NiftiImageDataset(t1_directory, clinical_data, patient_key, sub_transforms, slices)
        self.t2_dataset = S3NiftiImageDataset(t2_directory, clinical_data, patient_key, sub_transforms, slices)

        self.t1_patients = self.t1_dataset.patients
        self.t2_patients = self.t2_dataset.patients

        # Get IDs common across t1 and t2 datasets
        self.patients = list(set(self.t1_patients).intersection(self.t2_patients))

    def __getitem__(self, index):
        t1_image, t1_labels = self.t1_dataset.getDataByUID(self.uids[index])
        t2_image, t2_labels = self.t2_dataset.getDataByUID(self.uids[index])

         # If you're running into these assertion errors, something is very wrong!
        assert torch.all(t1_labels == t2_labels), 'Label mismatch when loading T1 and T2 images for patient {}'.format(self.uids[index])

        # Now that we know the labels are equal, we can take either and assign them to the patient labels
        labels = t1_labels

        # Concatenate the images along the channel dimensions
        # You can run into issues here if they are not the same size
        image = torch.cat((t1_image, t2_image), 0)

        if self.transforms is not None:
            return self.transforms(image), labels
        return image, labels
    
    def __len__(self):
        return len(self.patients)

class T1T2SurvivalDataset(S3ImageDataset):

    '''
    This dataset takes a S3NiftiImageDataset for t1 and t2 images

    It will load the t1 and t2 images, stack them along the channel dimension, then apply augmentations to the joint 2-channel image

    Identical to T1T2ImageDataset - but assumes labels are for survival analysis
    '''

    def __init__(self, t1_directory, t2_directory, clinical_data, patient_key, slices= False, transforms=None):

        # We'll initialize the parent class here using T1 data so we have access to the functions present.
        # Where needed, we'll overload member variables to account for the inclusion of t2 images (particularly, self.patients)
        # If we expand on this later, it would be useful to create a new base class that does not assume one image modality
        # mainly for cleanliness/maintainability, this works fine for now
        super().__init__(t1_directory, patient_key)
        self.t1_dataset = None
        self.t2_dataset = None
        self.transforms = transforms

        sub_transforms = Compose([
            EnsureChannelFirst(channel_dim=0),
            Resize(spatial_size=(64,64,64))
        ])

        # It seems a little disjointed to apply separate transforms (augmentations) to the individual t1/t2 images before they are concatenated
        # We'll initialize our sub-datasets with only transforms required to concatenate them without issue, 
        # We'll apply the rest of the transforms in the __getitem__ function for this class
        # That way, both the T1 and T2 images can be concatenated without issue while still receiving the same augmentations

        self.t1_dataset = S3NiftiSurvivalDataset(t1_directory, clinical_data, patient_key, sub_transforms, slices)
        self.t2_dataset = S3NiftiSurvivalDataset(t2_directory, clinical_data, patient_key, sub_transforms, slices)

        self.t1_patients = self.t1_dataset.patients
        self.t2_patients = self.t2_dataset.patients

        # Get IDs common across t1 and t2 datasets
        self.patients = list(set(self.t1_patients).intersection(self.t2_patients))

    def __getitem__(self, index):
        t1_image, t1_events, t1_durations = self.t1_dataset.getDataByUID(self.uids[index])
        t2_image, t2_events, t2_durations = self.t2_dataset.getDataByUID(self.uids[index])

        # If you're running into these assertion errors, something is very wrong!
        assert torch.all(t1_events == t2_events), 'Label mismatch when loading T1 and T2 images for patient {}'.format(self.uids[index])
        assert torch.all(t1_durations == t2_durations), 'Label mismatch when loading T1 and T2 images for patient {}'.format(self.uids[index])

        # Now that we know the labels are equal, we can take either and assign them to the patient labels

        events = t1_events
        durations = t1_durations

        # Concatenate the images along the channel dimensions
        # You can run into issues here if they are not the same size
        image = torch.cat((t1_image, t2_image), 0)

        if self.transforms is not None:
            return self.transforms(image), events, durations
        return image, events, durations

    def len(self):
        return len(self.patients)
    