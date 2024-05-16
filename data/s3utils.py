import SimpleITK as sitk
import pandas as pd
import tempfile
import numpy as np
import boto3
import nibabel as nib
import os
import botocore
from data.constants import *
from exceptions.exceptions import *

def readCSVS3(data_key):
    '''
    args: data_key - string containing path to csv file within the s3 bucket
    returns: Pandas dataframe containing the CSV data
    
    Reads clinical data from csv
    '''
    
    data_location = 's3://{}/{}'.format(BUCKET, data_key) 

    table = pd.read_csv(data_location)
    return table

def getDicomFilenames(prefix):
    '''
    args: prefix - string containing the prefix of search directory in the S3 bucket (ex. 'images/T1/Images' or 'images/T2/Masks')
    Returns: List of lists containing the dicom artifacts for each patient
    
    Parses the contents of the S3 bucket to provide filenames of dicom artifacts corresponding to each patient
    The outer list contains a list for each patient
    Each inner list will contain one string for each dicom artifact key corresponding to that patient
    '''
    
    filenames = []
    client = boto3.client('s3')
    keys = []
    all_keys = []
    
    # List all objects under search directory
    response = client.list_objects(Bucket=BUCKET, Prefix=prefix)
    
    s3r = boto3.resource('s3')
    bucket = s3r.Bucket(BUCKET)
    
    # List all the objects in the bucket
    files_in_bucket = list(bucket.objects.all())
    
    # Remove files that do not start with the given prefix
    trimmed_file_list = [x.key for x in files_in_bucket if x.key.startswith(prefix)]
    
    for key in trimmed_file_list:
        
        # Get the dicom directory of the patient the key corresponds to
        filename = key.split('/')[-2]
        
        # If this is a new patient, add the existing saved keys to the final list
        if filename not in filenames:
            filenames.append(filename)
            # Don't append an empty list when processing first patient
            if len(keys) > 0:
                all_keys.append(keys)
            keys = []
        
        # If this is not a new patient, add the key to the existing set of keys corresponding to the current patient 
        keys.append(key)
        
    # manually append keys for final patient
    all_keys.append(keys)
    return all_keys

def loadDicom(keys):
    '''
    args: keys - list of strings, each string is an s3 key for a .dcm file
                 The length of the list is determined by the number of slices in the patient MRI
                 Each slice will have an associated .dcm file
    returns: SimpleITK Image containing pixel data of patient MRI or Segmentation Mask
    
    Loads 3D image volume for a single patient from S3 bucket
    
    '''
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET)
    
    # Create a temporary directory on the compute node
    # We will store all .dcm files for each patient in the temporary directory to allow for easy reading by sitk
    with tempfile.TemporaryDirectory() as tmpdir:
        for key in keys:
            object = bucket.Object(key)
            
            # Sagemaker OS doesn't support filenames with separators
            trunc_key=key.split('/')[-1]
            with open('{}/{}'.format(tmpdir,trunc_key),'wb') as f:
                try:
                    object.download_fileobj(f)
                
                # This shouldn't be an issue, but some patients may be getting their keys parsed incorrectly
                # Print statements are for debugging purposes
                except Exception as e:
                    print('Could not find {}'.format(key))
                    raise e
        image = readSitk(tmpdir)
        
    return image

def parseS3ImageFolder(prefix):
    '''
    args:
        prefix - path to directory within s3 bucket containing list of patients with DICOM Image data
            See data.ImageDatasets.S3DicomDataset for expected directory structure
    
    returns:
        dcm_dict
            Keys - Anon MRNs for each patient in the ImageFolder specified by 'prefix'
            Values - dict containing two keys, 'image' and 'mask'
                'image' - list of strings, each string corresponds to a key for a single .dcm artifact associated with the patient (one key for each slice in image)
                'mask' - single element list containing string corresponding to the key of the mask associated with the patient
    '''


    s3r = boto3.resource('s3')
    bucket = s3r.Bucket(BUCKET)
    
    # List all the objects in the bucket
    files_in_bucket = list(bucket.objects.all())
    
    # Remove files that do not start with the given prefix
    trimmed_file_list = [x.key for x in files_in_bucket if x.key.startswith(prefix)]

    dcm_dict = {}
    for key in trimmed_file_list:

        # truncate the key to avoid string parsing issues resulting from unspecified paths in the s3 bucket 
        trunc_key = key.replace(prefix, '')

        # grab the anon MRN - if else allows agnosticism of trailing seperator in config
        anon_mrn = trunc_key.split(os.sep)[0] if trunc_key.split(os.sep)[0] != '' else trunc_key.split(os.sep)[1]
        
        # For the first key associated with a new patient, we need to initialize an entry for the patient to the dcm_dict
        if not anon_mrn in dcm_dict.keys():
            dcm_dict[anon_mrn] = {'image': [], 'mask': []}

        # If the key is a part of the image, add it to the 'image' key of the sub-dictionary for the patient
        if 'image' in trunc_key:
            dcm_dict[anon_mrn]['image'].append(key)

        # If the key points to the mask, add it to the 'mask' key of the subdictionary
        elif 'mask' in trunc_key:
            dcm_dict[anon_mrn]['mask'].append(key)
        
        else:
            raise InitializationError('Could not initialize S3ImageDataset - Unable to parse S3 key {}\n This is likely due to an issue with unexpected formatting of \
                                      the dataset on s3. See data.ImageDatasets.S3DicomDataset docstring for expected formatting of image data on S3'.format(key))
    return dcm_dict



def loadNifti(key):
    '''
    args: key - Path to nifti file in S3 bucket
    returns: Numpy Array containing pixel data of patient MRI or Segmentation Mask
    
    Loads 3D image volume for a single patient from S3 bucket
    
    '''
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(BUCKET)
    
    # Create a temporary directory on the compute node
    # We will store all .dcm files for each patient in the temporary directory to allow for easy reading by sitk
    with tempfile.TemporaryDirectory() as tmpdir:
        object = bucket.Object(key)
        
        # Sagemaker OS doesn't support filenames with separators
        trunc_key=key.split('/')[-1]
        with open('{}/{}'.format(tmpdir,trunc_key),'wb') as f:
            try:
                object.download_fileobj(f)
            
            # This shouldn't be an issue, but some patients may be getting their keys parsed incorrectly
            # Print statements are for debugging purposes
            except Exception as e:
                print('Could not find {}'.format(key))
                raise e
                
        image = nib.load('{}/{}'.format(tmpdir,trunc_key)).get_fdata()
        
    return image

def getNiftiFilenames(prefix):
    '''
    args: prefix - string containing the prefix of search directory in the S3 bucket (ex. 'n4corrected/T1' or 'n4corrected/T2')
    Returns: List of tuples containing the nifti paths for each patient
    
    Parses the contents of the S3 bucket to provide filenames of dicom artifacts corresponding to each patient
    The outer list contains a tuple for each patient
    Each tuple will contain be in the format (image_path, mask_path)
    '''
    
    filenames = []
    client = boto3.client('s3')
    all_keys = []
    
    # List all objects under search directory
    response = client.list_objects(Bucket=BUCKET, Prefix=prefix)
    
    s3r = boto3.resource('s3')
    bucket = s3r.Bucket(BUCKET)
    
    # List all the objects in the bucket
    files_in_bucket = list(bucket.objects.all())
    
    # Remove files that do not start with the given prefix
    trimmed_file_list = [x.key for x in files_in_bucket if x.key.startswith(prefix)]
    
    for key in trimmed_file_list:
        
        # Get the filename from the key
        filename = key.split('/')[-1]

        # If its an image, add the image and the mask to the final list (leverages naming convention)
        if filename.startswith('scan_'):
            keys = (key, key.replace('scan_', 'tumor_mask_'))
            all_keys.append(keys)
        
    return all_keys

def filenameToClinical(dicom_dir, patient_key):
    '''
    args: dicom_dir - string containing DICOM directory for an individual patient
          patient_key - Dataframe containing clinical variables and UIDs for each patient
    returns: Dataframe containing the clinical outcomes and UIDS for the provided patient
    
    Uses the patient key to lookup clinical variables for a patient given the filename of the DICOM directory
    '''
    # Find UID from a given filename
    uid = '-'.join(dicom_dir.split('-')[:2])

    # Look up UID in patient key -return corresponding row
    clinical_outcomes = patient_key.loc[patient_key['Anon Name'] == uid]

    return clinical_outcomes

def readSitk(dicom_dir):
    '''
    args: dicom_dir - String indicating directory where dicom data is stored
    returns: sitk Image corresponding to provided dicom directory
    
    Reads in series of dicom images from a directory, returns corresponding 3D volume
    '''
    
    reader = sitk.ImageSeriesReader()
    
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
        
    series_filenames = reader.GetGDCMSeriesFileNames(dicom_dir)
    reader.SetFileNames(series_filenames)

    image = reader.Execute()

    # sitk misinterprets 3D masks as 4D with one singleton dimension
    # If this is the case, reduce to 3D
    if image.GetDimension()==4 and image.GetSize()[3]==1:
        image = image[...,0]
        
    return image