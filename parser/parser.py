import yaml
import os
import logging

from models.resnet import r3d_18
from models.densenet import DenseNet121, TinyDensenet, TinyCNN
from models.multimodal import MultiModalModel
from monai.networks.nets import DenseNet121 as monaidensenet
from monai.networks.nets import Densenet201, SEResNet50
from exceptions.exceptions import *

from data.ClinicalDatasets import PreopClassificationDataset, PreopSurvivalDataset, PostopClassificationDataset, PostopSurvivalDataset
from data.ImageDatasets import ImageSurvivalDataset, ImageClassificationDataset, ImageSegmentationDataset, ImageDatasetByUIDs, \
    S3NiftiImageDataset, NiftiImageDataset, NiftiSurvivalDataset, S3DicomDataset, T1T2ImageDataset, T1T2SurvivalDataset
from data.RadiomicsDatasets import RadiomicsClassificationDataset, RadiomicsSurvivalDataset
from data.MultiModalDatasets import MultiModalDataset, MultiModalSurvivalDataset
import botocore

logger = logging.getLogger(__name__)

class Parser:

    '''
    This class is used to parse model & dataset configuration information.
    Instantiates relevent objects based on the configuration information provided.
    '''

    def __init__(self, config_path):
        self.config_path = config_path
        self.config = None

    def parseConfig(self):
        with open(self.config_path) as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise exc
            
        if self.config['ImageModel']['modality'].lower().startswith('t1t2') and self.config['ImageModel']['in_channels'] != 2:
            raise ConfigurationError('T1T2 ImageModel modality requires 2 input channels - current number of in_channels: {}'.format(self.config['ImageModel']['in_channels']))
        return self.config
    
    def getDatasets(self, args, image_path=None):
        datasets = []

        if args.classification:
            if args.preop:
                datasets.append(PreopClassificationDataset(self.config['Data']['data_loc']))
            elif args.postop:
                datasets.append(PostopClassificationDataset(self.config['Data']['data_loc']))
            
            if args.radiomics:
                datasets.append(RadiomicsClassificationDataset(self.config['Data']['rad_loc'], self.config['Data']['data_loc']))

            if args.images:

                if image_path is not None and type(image_path) == tuple:
                    datasets.append(T1T2ImageDataset(*image_path, self.config['Data']['data_loc'], self.config['Data']['key_loc']))               

                else:
                    # Check if we're on AWS
                    try:
                        datasets.append(S3NiftiImageDataset(image_path, self.config['Data']['data_loc'], self.config['Data']['key_loc']))
                    # Use our development set if we're running locally
                    except (botocore.exceptions.NoCredentialsError, botocore.exceptions.ClientError) as e:
                        datasets.append(ImageClassificationDataset(image_path, self.config['Data']['data_loc'], self.config['Data']['key_loc']))
        if args.survival:
            if args.preop:
                datasets.append(PreopSurvivalDataset(self.config['Data']['data_loc']))
            elif args.postop:
                datasets.append(PostopSurvivalDataset(self.config['Data']['data_loc']))
            
            if args.radiomics:
                datasets.append(RadiomicsSurvivalDataset(self.config['Data']['rad_loc'], self.config['Data']['data_loc']))

            if args.images:
                # Check for T1 & T2 combined image dataset
                if image_path is not None and type(image_path) == tuple:
                    datasets.append(T1T2SurvivalDataset(*image_path, self.config['Data']['data_loc'], self.config['Data']['key_loc']))
                # Otherwise just use the single dataset
                else:
                    datasets.append(NiftiSurvivalDataset(image_path, self.config['Data']['data_loc'], self.config['Data']['key_loc']))

        if args.segmentation:
            datasets.append(ImageSegmentationDataset(self.config['Data']['image_loc']))

        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            if args.classification:
                dataset = MultiModalDataset(datasets)
            elif args.survival:
                dataset = MultiModalSurvivalDataset(datasets)
            else:
                raise ConfigurationError('Could not determine multimodal dataset type - specify one of --survival or --classification')
            
        return dataset
    
    def getModel(self, args):
        model = None

        if self.config is None:
            raise InitializationError('Attempted to load model prior to parsing config parameters, config must be parsed prior to loading model')
        
        model_name = self.config['ImageModel']['name'].lower()
        if model_name.startswith('densenet121'):
            model = DenseNet121(
                spatial_dims=self.config['ImageModel']['spatial_dims'],
                in_channels=self.config['ImageModel']['in_channels'],
                out_channels=self.config['ImageModel']['num_classes'],
                feature_channels=self.config['ImageModel']['feature_layers'],
                dropout_prob=self.config['ImageModel']['dropout_prob']
            )

        elif model_name.startswith('tinydensenet'):
            model = TinyDensenet(
                spatial_dims=self.config['ImageModel']['spatial_dims'],
                in_channels=self.config['ImageModel']['in_channels'],
                out_channels=self.config['ImageModel']['num_classes'],
                feature_channels=self.config['ImageModel']['feature_layers'],
                dropout_prob=self.config['ImageModel']['dropout_prob']
            )

        elif model_name.startswith('tinycnn'):
            model = TinyCNN(
                in_channels=self.config['ImageModel']['in_channels'],
                out_channels=self.config['ImageModel']['num_classes'],
                feature_channels=self.config['ImageModel']['feature_layers'],
                dropout_prob=self.config['ImageModel']['dropout_prob']
            )
        elif model_name.startswith('monaidensenet'):
            model = monaidensenet(
                spatial_dims=self.config['ImageModel']['spatial_dims'],
                in_channels=self.config['ImageModel']['in_channels'],
                out_channels=self.config['ImageModel']['num_classes'],
                dropout_prob=self.config['ImageModel']['dropout_prob']
            )
        elif model_name.startswith('densenet201'):

            model = Densenet201(
                spatial_dims=self.config['ImageModel']['spatial_dims'],
                in_channels=self.config['ImageModel']['in_channels'],
                out_channels=self.config['ImageModel']['num_classes'],
            )
        elif model_name.startswith('seresnet50'):
            model = SEResNet50(
                spatial_dims=self.config['ImageModel']['spatial_dims'],
                in_channels=self.config['ImageModel']['in_channels'],
                out_channels=self.config['ImageModel']['num_classes'],
            )
        elif model_name.startswith('r3d_18'):
            model = r3d_18(self.config['ImageModel']['num_classes'])
        else:
            raise ConfigurationError('Model name not recognized: {}\n\tSee config file for valid options'.format(model_name))
        
        logger.info('Successfully loaded configured model: {}'.format(model_name))
        
        if args.images and args.preop:
            assert model_name.startswith('tinycnn') or model_name.startswith('tinydensenet') or model_name.startswith('densenet121'), \
            'Image models used to build multimodal models must be one of \'tinycnn\', \'tinydensenet\' or \'densenet121\''

            model = MultiModalModel(
                model, 
                self.config['ClinicalModel']['PRE_OP_PREDICTORS'],
                self.config['ImageModel']['num_classes'], 
                self.config['ImageModel']['feature_layers'],
                blend = args.blend
            )

        elif args.images and args.postop:
            assert model_name.startswith('tinycnn') or model_name.startswith('tinydensenet') or model_name.startswith('densenet121'), \
            'Image models used to build multimodal models must be one of \'tinycnn\', \'tinydensenet\' or \'densenet121\''

            model = MultiModalModel(
                model, 
                self.config['ClinicalModel']['PRE_OP_PREDICTORS'] + self.config['ClinicalModel']['POST_OP_PREDICTORS'],
                self.config['ImageModel']['num_classes'], 
                self.config['ImageModel']['feature_layers'],
                blend=args.blend
            )

        return model
    
    def getImagePath(self):
        modality = self.config['ImageModel']['modality'].lower()
        image_path = None
        if modality.startswith('t1t2'):
            t1_path = os.path.join(self.config['Data']['image_loc'], self.config['Data']['t1_path'])
            t2_path = os.path.join(self.config['Data']['image_loc'], self.config['Data']['t2_path'])
            image_path = (t1_path, t2_path)
        elif modality.startswith('t1'):
            image_path = os.path.join(self.config['Data']['image_loc'], self.config['Data']['t1_path'])
        elif modality.startswith('t2'):
            image_path = os.path.join(self.config['Data']['image_loc'], self.config['Data']['t2_path'])
        else:
            raise ConfigurationError('Could not recognize requested Image Modality {} \n Options are \'t1\', \'t2\', or \'t1t2\''.format(self.config['ImageModel']['modality']))
        
        return image_path