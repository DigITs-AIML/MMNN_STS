import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from monai.transforms import \
    Compose, Resize, ScaleIntensity, ToTensor, RandRotate, RandFlip, RandZoom, EnsureChannelFirst
import sys
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 600000000 # need to increase the maximum allowable pixels, or PIL will suspect WSIs are malicious
from PIL import Image
from PIL.TiffTags import TAGS


from data.constants import RADIOMICS_UID
from data.utils import loadImage, loadMask, anonIDToRadiomicsUID, tileImage, feature_extract_tiles
from data.s3utils import getNiftiFilenames, getPathologyFilenames, loadTiff, loadNifti, readCSVS3, parseS3ImageFolder, loadDicom
from exceptions.exceptions import *
from data.encoder import resnet50
from utils.utils import loadUIDs

# ALL_UIDS = loadUIDs('train_uids.txt') + loadUIDs('val_uids.txt')
ALL_UIDS = loadUIDs('test_uids.txt')

class PathologyDataset(Dataset):
    def __init__(self, patients, patient_key):

         #TODO : Align expected preprocessing of anon MRNs (self.patients) BEFORE initializing parent class between s3 nifti and dicom datasets
        # Read patient key
        self.patient_key = readCSVS3(patient_key)

        path_uids = [x.split(os.sep)[-2] for x in patients]

        self.patients = self.patient_key.set_index('Pathology Accession Number').loc[path_uids, 'XNAT Anon Name'].tolist()

        # self.patients = self.patient_key.loc[patient_key['Pathology Accession Number'] == path_uids]['XNAT Anon Name']
        print(self.patients)

        self.multimodal_identifier = 'wsi'

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.patients)

    def getDataByUID(self, uid):
        assert self.patient_key is not None, 'Must provide key to map anonymized ids to MRNs'
        patient_ids = self.patient_key.loc[self.patient_key[RADIOMICS_UID] == uid]
        anon_id = patient_ids['XNAT Anon Name'].item()

        index = self.patients.index(anon_id)
        return self.__getitem__(index)
    
    @property
    def uids(self):
        assert self.patient_key is not None, 'Must provide key to map anonymized ids to MRNs'
        uids = []
        for anon_id in self.patients:
            patient_ids = self.patient_key.loc[self.patient_key['XNAT Anon Name'] == anon_id] # Old keys use 'Anon Name'
            mrn = patient_ids[RADIOMICS_UID].item()
            uids.append(int(mrn))
        return uids


class S3WSISurvivalDataset(PathologyDataset):
    def __init__(self, prefix, clinical_data, patient_key, tile=True, transforms=None):
        patients = getPathologyFilenames(prefix)
        super().__init__(patients, patient_key)
        self.patient_filenames = patients
        # self.survival_dataset = PreopSurvivalDataset(clinical_data)
        self.transforms = transforms
        self.prefix = prefix.split(os.sep)[-1]
        self.tile_size = 512
        self.tile = tile
        

    def __getitem__(self, index):

        key = self.patient_filenames[index]

        uid = self.uids[index]
        if uid not in ALL_UIDS:
            return None, None
        
        image, meta_dict = loadTiff(key)

        uid = np.float64(uid)
        # mrns = self.survival_dataset.targets[RADIOMICS_UID]

        

        if self.tile:
            features = []
            tiles = tileImage(image, meta_dict, self.tile_size)

        return uid, tiles



    def __len__(self):
        return len(self.patients)
    

def main():

    device = 'cuda:0'
    image_path = '/path/to/pathology_images'
    data_loc = 's3://s3_uri_of_clinical_data.csv'
    key_loc = 's3_key_of_patient_key' #TODO: update to use URI

    encoder = resnet50(pretrained=True).to(device)
    dataset = S3WSISurvivalDataset(image_path, data_loc, key_loc)

    print(len(dataset))

    max_num_tiles = 0 #pocket algorithm to find maximum number of tiles in dataset

    for uid, tiles in dataset:

        if uid is None:
            continue

        filename = './path_features/{}.txt'.format(uid)

        if len(tiles) > max_num_tiles:
            max_num_tiles = len(tiles)

        features = []
        for tile in tiles:
            tile = torch.tensor(tile[None, ...], dtype=torch.float32)
            tile = tile.to(device)

            tile_features = encoder(tile)

            tile_features = list(tile_features.cpu().flatten())
            tile_features = [x.item() for x in tile_features]

            features = features + tile_features

        with open(filename, 'w') as f:
            f.write('\n'.join([str(x) for x in features]))

        # Make sure we can read it
        with open(filename, 'r') as f:
            features = [float(line.strip()) for line in f]
            

        print('max_num_tiles: ' + str(max_num_tiles))
        

if __name__ == '__main__':
    main()