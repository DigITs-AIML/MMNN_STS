import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, datasets, transforms=None):
        self.datasets = datasets
        self._transforms = transforms
        all_mrns = [dataset.uids for dataset in self.datasets]

        # Get common uids between all provided datasets
        self.mrns = list(set.intersection(*map(set, all_mrns)))

    def __getitem__(self, index):
        mrn = self.mrns[index]
        data = {}
        target = None
        for dataset in self.datasets:
            new_data,new_target = dataset.getDataByUID(mrn)
            data[dataset.multimodal_identifier] = new_data
            if target is not None:
                assert all(new_target == target), 'Disimilar target variables between one or more of the provided datasets: patient {}'.format(mrn)
            else:
                target = new_target
            
        return data, target

    def __len__(self):
        return len(self.mrns)

    def getDataByUID(self, uid):
        index = self.mrns.index(uid)
        return self.__getitem__(index)
    
    @property
    def uids(self):
        return self.mrns
    
    @property
    def transforms(self):
        return self._transforms
    
    @transforms.setter
    def transforms(self, transforms):
        self._transforms = transforms
        # Find the image dataset and set it's transforms appropriately
        # TODO: We're kind of fighting through a few layers of abstraction to apply these augmentations correctly, perhaps there is a better way
        for i, dataset in enumerate(self.datasets):
            if dataset.multimodal_identifier == 'image':
                self.datasets[i].transforms = transforms
    @property
    def clinical_dataset(self):
        for dataset in self.datasets:
            if dataset.multimodal_identifier == 'clinical':
                return dataset
        raise ValueError('Attempted to retreive a clinical dataset when no dataset has a \'clinical\' multimodal identifier')
    
class MultiModalSurvivalDataset(MultiModalDataset):

    def __init__(self, datasets, transforms = None):
        super().__init__(datasets, transforms=transforms)
        for dataset in self.datasets:
            print(type(dataset))
    
    def __getitem__(self, index):
        mrn = self.mrns[index]
        data = {}
        event = None
        duration = None

        for dataset in self.datasets:
            new_data,new_event, new_duration = dataset.getDataByUID(mrn)
            data[dataset.multimodal_identifier] = new_data

            if event is not None or duration is not None:
                assert torch.all(new_event == event) and torch.all(new_duration == duration), 'Disimilar target variables between one or more of the provided datasets: patient {}'.format(mrn)
            else:
                event = new_event
                duration = new_duration
            
        return data, event, duration


        