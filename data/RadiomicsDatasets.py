import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data.constants import RADIOMICS_UID
from data.utils import loadRadiomicsData
from data.ClinicalDatasets import PreopSurvivalDataset, PreopClassificationDataset

class RadiomicsDataset(Dataset):
    def __init__(self, radiomics_path):
        self.data, self.targets = loadRadiomicsData(radiomics_path)
        self.mrns = self.targets[RADIOMICS_UID].tolist()

    def __getitem__(self, index):
        pass

    def __len__(self):
        return self.data.shape[0]

    def getDataByUID(self, uid):
        index = self.mrns.index(uid)
        return self.__getitem__(index)
    
    @property
    def uids(self):
        return [int(x) for x in self.mrns]

class RadiomicsSurvivalDataset(RadiomicsDataset):

    def __init__(self, radiomics_path, clinical_data):
        super().__init__(radiomics_path)

        #preop or postop doesn't matter - we just want the targets
        self.survival_dataset = PreopSurvivalDataset(clinical_data)
        
    
    def __getitem__(self, index):

        #data
        data = self.data.iloc[index].tolist()

        #label
        _, label = self.survival_dataset.getDataByUID(self.mrns[index])

        return data, label

class RadiomicsClassificationDataset(RadiomicsDataset):

    def __init__(self, radiomics_path, clinical_data):
        super().__init__(radiomics_path)
        self.classification_dataset = PreopClassificationDataset(clinical_data)

    def __getitem__(self, index):
        #data
        data = self.data.iloc[index].tolist()
        _, label = self.classification_dataset.getDataByUID(self.mrns[index])
        return data, label