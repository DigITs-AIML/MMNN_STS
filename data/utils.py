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
    table = pd.read_csv(file, usecols=PRE_OP_PREDICTORS+POST_OP_PREDICTORS+TARGETS_BINARY+TARGETS_TIME+[RADIOMICS_UID], nrows=NUMROWS)
    return table

def loadClinical(filename):
    data = _load_clinical(filename)

    # Convert string dates to date times
    data[[x for x in TARGETS_TIME[1:]]] = data[[x for x in TARGETS_TIME[1:]]].apply(pd.to_datetime)

    # Convert Date of LR/Distant Metastasis/Reop to # of days from surgery
    # Time_LR = data[TARGETS_TIME[1]] - data[TARGETS_TIME[-1]]
    Time_MET = data[TARGETS_TIME[1]] - data[TARGETS_TIME[-1]]
    # Time_REOP = data[TARGETS_TIME[3]] - data[TARGETS_TIME[-1]]

    # Remove the date columns
    for x in TARGETS_TIME[1:]:
        data = data.drop(x, axis=1)

    # Add the columns for # days since surgery
    # data['Time_LR'] = Time_LR.apply(lambda x: x.days)
    data['Time_MET'] = Time_MET.apply(lambda x: x.days)
    # data['Time_REOP'] = Time_REOP.apply(lambda x: x.days)

    # Convert categorical columns from strings to indexed valuues
    for header in HEADERS_TO_CONVERT:
        data = convert_to_index(data, header)

    # hacky reformatting for necrosis column to remove % signs and be able to convert to float type
    # data['Necrosis % (information not known prior to operation)'] = data['Necrosis % (information not known prior to operation)'].replace(to_replace=np.nan, value='-1%')
    # data['Necrosis % (information not known prior to operation)'] = data['Necrosis % (information not known prior to operation)'].apply(lambda x: str(x)[:-1])
    # data['Necrosis % (information not known prior to operation)'] = data['Necrosis % (information not known prior to operation)'].replace(to_replace='-1', value=np.nan)

    # mix of strings, ints, and floats in dataframe - convert all to floats
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

        print(time_header)

        print(data[time_header])

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

    print(full_targets.keys())

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
    data.loc[data['Tumor Volume (cm^3) (MRI)'] < 500, 'VolumeCutoff'] = 0
    data.loc[data['Tumor Volume (cm^3) (MRI)'] >= 500, 'VolumeCutoff'] = 1
    data.loc[data['Tumor Volume (cm^3) (MRI)'] > 1000, 'VolumeCutoff' ] = 2

    return data


## Image tiling
def convert_to_uint8(image, dtype=np.uint8, min_val=0, max_val=255):
    type_ = (str(dtype)).lower()
    if 'uint' in type_:
        image = np.array(image, dtype=np.uint8)
    else:
        image = np.array((image-min_val)/(max_val-min_val)*255, dtype=np.uint8)
    
    return image

def convert_from_uint8(image, dtype=np.uint8, min_val=0, max_val=255):
    type_ = (str(dtype)).lower()
    if 'float' in type_:
        image = np.array(image/255*(max_val-min_val)+min_val, dtype=dtype)
    elif 'uint8' in type_:
        image = np.array(image, dtype=np.uint8)
    else:
        image = np.array(round(image/255*(max_val-min_val)+min_val), dtype=dtype)
        
    return image

class SplitImage:
    """
    Class to split an image into tiles based on a grid (m x n) and overlap between tiles
    """
    def __init__(self, image, grid, overlap):
        """
        Initialize the SplitImage class.

        Args:
            image (ndarray): Input image 
            grid (tuple): Grid dimensions (rows, columns) to split the image.
            overlap (int): Number of pixels to overlap between tiles.
        """
        # self.type, self.min_val, self.max_val = what_is(image)
        self.type = np.uint8
        self.min_val = 0
        self.max_val = 255
        self.image = convert_to_uint8(image, dtype=self.type, min_val=self.min_val, max_val=self.max_val)
        self.grid = grid
        self.overlap = max(overlap, 1)
        
        # print("\nImage channels: ", image.shape[2] if len(image.shape) == 3 else 1)
        # print("Input data type: ", type)
  
    
    def pad_image(self, image, padding):
        """
        Pad the image with zeros.

        Args:
            image (ndarray): Input image.
            padding (tuple): Padding values for top, bottom, left, and right.

        Returns:
            ndarray: Padded image.
        """
        if len(image.shape) == 2:
            image = np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3])), 'constant', constant_values=0)
        else:
            image = np.pad(image, ((padding[0], padding[1]), (padding[2], padding[3]), (0, 0)), 'constant', constant_values=0)

        return image

    def check_divisible(self, image):
        """
        Check if the image dimensions are divisible by the grid dimensions.
        If not, pad the image to make it divisible.

        Args:
            image (ndarray): Input image.

        Returns:
            ndarray: Image with dimensions divisible by the grid.
        """
        h, w = image.shape[:2]
        w_pad = (self.grid[1] - (w % self.grid[1])) % self.grid[1]
        h_pad = (self.grid[0] - (h % self.grid[0])) % self.grid[0]

        new_image = self.pad_image(image, (0, h_pad, 0, w_pad))

        return new_image

    def extract_tile(self, image, tile_height, tile_width, i, j):
        """
        Extract a tile from the image.

        Args:
            image (ndarray): Input image.
            tile_height (int): Height of the tile.
            tile_width (int): Width of the tile.
            i (int): Row index of the tile.
            j (int): Column index of the tile.

        Returns:
            ndarray: Extracted tile.
        """
        overlap = self.overlap
        x1 = i * tile_height
        x2 = (i + 1) * tile_height + 2 * overlap

        y1 = j * tile_width
        y2 = (j + 1) * tile_width + 2 * overlap

        if len(image.shape) == 2:
            tile = image[x1:x2, y1:y2]
        else:
            tile = image[x1:x2, y1:y2, :]
        return tile

    def split_image(self, show_rect=True, show_tiles=True):
        """
        Split the image into multiple tiles.

        Args:
            show_rect (bool, optional): Whether to draw rectangles on the tiles. Defaults to True.
            show_tiles (bool, optional): Whether to display the tiles. Defaults to True.

        Returns:
            list: List of tiles.
        """
        overlap = self.overlap
        image = self.check_divisible(self.image)

        height, width = image.shape[:2]
        tile_height = height // self.grid[0]
        tile_width = width // self.grid[1]

        image = self.pad_image(image, (overlap, overlap, overlap, overlap))
        c = image.shape[2] if len(image.shape) == 3 else 1
        
        tiles = []
        for i in range(self.grid[0]):
            for j in range(self.grid[1]):
                tile = self.extract_tile(image, tile_height, tile_width, i, j)
            
                # preview if c <= 3 else  do nothing
                if c <= 3: # if image is RGB or grayscale
                    cv2.rectangle(tile, (overlap, overlap), (tile_width + overlap, tile_height + overlap), (255, 0, 0), 1) if show_rect else None
                    
                tile = convert_from_uint8(tile, dtype=self.type, min_val=self.min_val, max_val=self.max_val)
                tiles.append(tile)

        self.preview_tiles(tiles) if show_tiles else None

        return tiles 

    def preview_tiles(self, tiles):
        """
        Display the tiles in a grid.

        Args:
            tiles (list): List of tiles of ndarray type.
        """
        grid = self.grid
        fig, ax = plt.subplots(grid[0], grid[1], figsize=(20, 20))
        if np.any(np.array(grid) == 1):
            ax = ax.flatten()
        for i, tile in enumerate(tiles):
            tile = convert_to_uint8(tile, dtype=self.type, min_val=self.min_val, max_val=self.max_val) # convert to uint8 for preview
            c = tiles[0].shape[2] if len(tiles[0].shape) == 3 else 1
            if c>3: # get the first channel of the image
                tile = tile[:, :, 0]
                plt.set_cmap('gray')
            else: # if image is RGB or grayscale
                tile = tile[:, :, ::-1] if len(tile.shape) == 3 else tile
                plt.set_cmap('gray') if len(tile.shape) == 2 else None
                
            if np.any(np.array(grid) == 1):
                ax[i].imshow(tile)
                ax[i].set_title('Tile {}'.format(i + 1))
            else:
                ax[i // grid[1], i % grid[1]].imshow(tile)
                ax[i // grid[1], i % grid[1]].set_title('Tile {}'.format(i + 1))
        plt.show()

def convert_to_uint8(image, dtype=np.uint8, min_val=0, max_val=255):
    type_ = (str(dtype)).lower()
    if 'uint' in type_:
        image = np.array(image, dtype=np.uint8)
    else:
        image = np.array((image-min_val)/(max_val-min_val)*255, dtype=np.uint8)
    
    return image

def tileImage(img, meta_dict, tile_size):
    img_width = meta_dict['ImageWidth'][0]
    img_height = meta_dict['ImageLength'][0]

    num_tiles_x = (img_width // tile_size) - 1
    num_tiles_y = (img_height // tile_size) - 1
    tiles = []
    img = convert_to_uint8(img)
    for x in range(num_tiles_x):
        for y in range(num_tiles_y):
            tile = img[x*tile_size : (x+1) * tile_size, y*tile_size : (y+1) * tile_size, :3]
            tile = np.moveaxis(tile, -1, 0)

            # Skips misshapen tiles (this shouldn't happen)
            if tile.shape[1] == tile_size and tile.shape[2] == tile_size:
                tiles.append(tile)
            else:
                continue
    print(len(tiles))
    print('finished tiling')
    return tiles


def feature_extract_tiles(model, tile):
    return model(tile)
    