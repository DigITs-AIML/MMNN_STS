import torch
from torch.utils.data import Dataset
from data.utils import getPreopBinary, getPostopBinary, getPreopSurvival, getPostopSurvival
from data.constants import RADIOMICS_UID

class ClinicalDataset(Dataset):
    def __init__(self, filename,preop=False, postop=False, classification=False, survival=False):
        assert preop or postop, 'Must specificy preop or postop data'
        assert classification or survival, 'Must specify classification or survival'
        assert not (preop and postop), 'May only specificy one of preop and postop'
        assert not (classification and survival), 'May only specify one of classification and survival'

        self.preop = preop
        self.postop = postop
        self.classification = classification
        self.survival = survival

        self.data, self.targets = None, None

        self.multimodal_identifier = 'clinical'

        if self.preop:
            if self.classification:
                self.data, self.targets = getPreopBinary(filename, include_uids=True)
            else:
                self.data, self.targets = getPreopSurvival(filename, include_uids=True)
        else:
            if self.classification:
                self.data, self.targets = getPostopBinary(filename, include_uids=True)
            else:
                self.data, self.targets = getPostopSurvival(filename, include_uids=True)

        assert self.data is not None and self.targets is not None, 'Could not assign either the data or targets'

    def __getitem__(self, index):

        data = self.data.iloc[index]

        if self.classification:
            target = self.targets.iloc[index]

            target=target.drop([RADIOMICS_UID])
            return torch.tensor(data), torch.tensor(target.to_numpy())
        elif self.survival:
            target = {}
            for key in self.targets.keys():
                if key == RADIOMICS_UID:
                    continue
                val = self.targets[key][index]
                target[key] = val
            
            events = [int(target[key][0]) for key in target.keys()]
            durations = [target[key][1] for key in target.keys()]
            
            return torch.tensor(data), torch.tensor(events), torch.tensor(durations)

    def __len__(self):
        return self.data.shape[0]

    def getDataByUID(self, uid):
        if self.classification:
            index = self.targets.loc[self.targets[RADIOMICS_UID] == uid].index.item()
        elif self.survival:
            index = self.targets[RADIOMICS_UID].loc[self.targets[RADIOMICS_UID] == uid].index.item()
        return self.__getitem__(index)

    @property
    def uids(self):
        return [int(x) for x in self.targets[RADIOMICS_UID].tolist()]

class PreopClassificationDataset(ClinicalDataset):
    
    def __init__(self, filename):
        super().__init__(filename, preop=True, classification=True)

class PreopSurvivalDataset(ClinicalDataset):
    
    def __init__(self, filename):
        super().__init__(filename, preop=True, survival=True)

class PostopClassificationDataset(ClinicalDataset):
    
    def __init__(self, filename):
        super().__init__(filename, postop=True, classification=True)

class PostopSurvivalDataset(ClinicalDataset):
    
    def __init__(self, filename):
        super().__init__(filename, postop=True, survival=True)