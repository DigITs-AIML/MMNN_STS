import SimpleITK as sitk
import pandas as pd
import numpy as np
from skmultilearn.model_selection.iterative_stratification import iterative_train_test_split
from data.constants import *


def anonIDToRadiomicsUID(anon_id, patient_key):
    # Find UID from a given filename

    # Look up UID in patient key -return corresponding row
    clinical_outcomes = patient_key.loc[patient_key['Anon MRN'] == anon_id]

    return clinical_outcomes[RADIOMICS_UID].item()

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

def convert_to_index(dataframe, header):
    # Converts string categorical columns to indexed categorical columns
    # For instance - a column with values 'Thigh', 'Pelvis', 'Arm' is converted to a column with values 0, 1, 2
    unique_values = list(dataframe[header].unique())

    for value in unique_values:
        dataframe[header] = dataframe[header].replace(to_replace=value, value=unique_values.index(value))
    return dataframe

def _load_clinical(file):
    table = pd.read_csv(file, usecols=PRE_OP_PREDICTORS+POST_OP_PREDICTORS+TARGETS_BINARY+TARGETS_TIME+[RADIOMICS_UID])
    return table

def loadClinical(filename):
    data = _load_clinical(filename)

    # Convert string dates to date times
    data[[x for x in TARGETS_TIME[1:]]] = data[[x for x in TARGETS_TIME[1:]]].apply(pd.to_datetime)

    # Convert Date of event to # of days from surgery
    Time_MET = data[TARGETS_TIME[1]] - data[TARGETS_TIME[-1]]

    # Remove the date columns
    for x in TARGETS_TIME[1:]:
        data = data.drop(x, axis=1)

    # Add the columns for # days since surgery
    data['Time_MET'] = Time_MET.apply(lambda x: x.days)

    # Convert categorical columns from strings to indexed valuues
    for header in HEADERS_TO_CONVERT:
        data = convert_to_index(data, header)

    # hacky reformatting for necrosis column to remove % signs and be able to convert to float type
    data['Necrosis % (information not known prior to operation)'] = data['Necrosis % (information not known prior to operation)'].replace(to_replace=np.nan, value='-1%')
    data['Necrosis % (information not known prior to operation)'] = data['Necrosis % (information not known prior to operation)'].apply(lambda x: str(x)[:-1])
    data['Necrosis % (information not known prior to operation)'] = data['Necrosis % (information not known prior to operation)'].replace(to_replace='-1', value=np.nan)

    # mix of numeric strings, ints, and floats in dataframe - convert all to floats
    data = data.astype(float)

    # Data is cleaned now

    return data

def loadRadiomicsData(radiomics_file):
    table = pd.read_csv(radiomics_file)

    for column in RADIOMICS_EXCLUDE_COLUMNS:
        table = table.drop(column, axis=1)
    
    labels = table[RADIOMICS_LABEL_COLUMNS + [RADIOMICS_UID]].copy()
    for column in RADIOMICS_LABEL_COLUMNS:
        table = table.drop(column, axis=1)
    
    return table.astype(float), labels.astype(int)

def getSurvTargetData(data, include_uids=True):
    # Target data in form np.array([(Uncensored_bool, surv time), (uncensored_bool, surv time), .... ])
    # uncensored bool = 1 if the data has not been censored
    # 0 if the data is censored
    # Survival time is actual survival time for uncensored data
    # Survival time is FUtime if the data is censored

    final_dict = {}

    if include_uids:
        final_dict[RADIOMICS_UID] = data[RADIOMICS_UID]

    for boolean_header, time_header in HEADER_PAIRS:

        # replace all nan values with the value in the FUtime column
        data[time_header].fillna(data.FUtime, inplace=True)

        data[boolean_header] = data[boolean_header].astype(bool)
        data[time_header] = data[time_header].astype(int)
        # zip the booleans and time variables into a list of tuples (pairs)
        survival_data = list(zip(data[boolean_header].tolist(), data[time_header].tolist()))

        # Convert to numpy array
        final_dict[time_header] = survival_data #np.array(survival_data, dtype=[('Status', '?'), ('Survival_in_days', '<f8')])

    # Final dict keys are 'FUtime', 'Time_LR', 'Time_MET', 'Time_REOP' 
    return final_dict

def getPreopBinary(filename,include_uids=False):
    data = loadClinical(filename)
    if include_uids:
        return data[PRE_OP_PREDICTORS], data[[RADIOMICS_UID] + TARGETS_BINARY]
    return data[PRE_OP_PREDICTORS], data[TARGETS_BINARY]

def getPostopBinary(filename, include_uids=False):
    data = loadClinical(filename)
    if include_uids:
        return data[POST_OP_PREDICTORS], data[[RADIOMICS_UID] + TARGETS_BINARY]
    return data[POST_OP_PREDICTORS], data[TARGETS_BINARY]

def getPreopSurvival(filename, include_uids):
    data = loadClinical(filename)
    data.fillna(-1)
    surv_targets = getSurvTargetData(data,include_uids=include_uids)
    return data[PRE_OP_PREDICTORS], surv_targets

def getPostopSurvival(filename, include_uids):
    data = loadClinical(filename)
    data.fillna(-1)
    surv_targets = getSurvTargetData(data,include_uids=include_uids)
    return data[POST_OP_PREDICTORS], surv_targets

def _stratifiedSplit(dataset, full_targets, uids, cutoffs=True):
    # copy the dataset so we don't modify the original
    dataset = dataset.copy()

    # Reduce the clinical data to just uids in original dataset
    targets = full_targets.loc[full_targets[RADIOMICS_UID].isin(uids)]
    dataset = dataset.loc[full_targets[RADIOMICS_UID].isin(uids)]
    
    # convert uids to numpy array - add an extra dimension or scikit learn will complain
    uids = np.array(uids)[..., None]

    # add cutoff variables for STS use case
    if cutoffs:
        dataset = add_cutoffs(dataset) 

    #  get stratification booleans from each source (data + targets), and concatenate
    data_stratify = dataset[STRATIFY_BY].to_numpy()
    target_stratify = targets[TARGETS_BINARY].to_numpy()

    targets = np.concatenate((data_stratify, target_stratify), axis=1)

    # do a 70/30 split, then a 50/50 split to get a 70/15/15 split
    train_uids, _, testval_uids, testval_targets = iterative_train_test_split(uids, targets, 0.3)
    val_uids, _, test_uids, _ = iterative_train_test_split(testval_uids, testval_targets, 0.5)
    train_uids, val_uids, test_uids = train_uids.squeeze().tolist(), val_uids.squeeze().tolist(), test_uids.squeeze().tolist()
    
    # Write out uids to files
    with open('train_uids.txt', 'w') as f:
        f.write('\n'.join([str(int(x)) for x in train_uids]))
    with open('val_uids.txt', 'w') as f:
        f.write('\n'.join([str(int(x)) for x in val_uids]))
    with open('test_uids.txt', 'w') as f:
        f.write('\n'.join([str(int(x)) for x in test_uids]))
    
    # return uids
    return train_uids, val_uids, test_uids

def add_cutoffs(data):
    volume_cutoff = [0] * data.shape[0]
    data['VolumeCutoff'] = volume_cutoff
    data.loc[data['TumorVolume (cm^3)'] < 500, 'VolumeCutoff'] = 0
    data.loc[data['TumorVolume (cm^3)'] >= 500, 'VolumeCutoff'] = 1
    data.loc[data['TumorVolume (cm^3)'] > 1000, 'VolumeCutoff' ] = 2

    return data