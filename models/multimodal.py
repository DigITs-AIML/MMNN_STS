import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from models.mlp import MLP
from utils.utils import FeatureExtractor, BackpropagatableFeatureExtractor, MultiModalGradCAM
from medcam import medcam
import sys
class MultiModalModel(nn.Module):
    def __init__(
        self,
        image_model,
        clinical_predictors,
        num_classes,
        num_features,
        blend=False
    ):
        """
        Args:
            image_model - image model
            clinical_predictors - list of clinical predictors
            num_classes - number of classes,
            num_features - number of features for each modality
        """
        super().__init__()
        self.image_model = image_model
        self.clinical_predictors = clinical_predictors
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_clinical_inputs = len(clinical_predictors)
        self.clinical_model = MLP(self.num_clinical_inputs, self.num_classes, self.num_features)
        # self.output_head = MLP(self.num_features * 2, self.num_classes)
        self.output_head = nn.Linear(self.num_features * 2, self.num_classes)
        self.blend=blend

        # Build clinical and image feature extractors

        # self.image_model = FeatureExtractor(self.image_model, ['features'])
        # self.clinical_model = FeatureExtractor(self.clinical_model, ['features'])

        self.image_model = BackpropagatableFeatureExtractor(self.image_model)
        self.clinical_model = BackpropagatableFeatureExtractor(self.clinical_model)

        # Output heads to be attached to image and clinical subnetworks when doing gradient blending
        # (Need a way to get predictions out of the modality subnetworks)
        self.clinical_output_head = nn.Linear(self.num_features, self.num_classes)
        self.image_output_head = nn.Linear(self.num_features, self.num_classes)



    def forward(self, x):
        
        # Assumes dictionary input with 'image' and 'clinical' keys
        image_data = x['image']
        clinical_data = x['clinical']

        # Forward pass of image subnetwork and clinical subnetwork
        image_features = self.image_model(image_data)
        clinical_features = self.clinical_model(clinical_data)

        # Concatenate features from each subnetwork
        features = torch.cat([image_features, clinical_features], 1)

        # send concatenated features through a final multimodal output head
        out = self.output_head(features)

        # If gradient blending - we also need the predictions made by each individual modality
        # To get this, we send features from each subnetwork into the corresponding output head
        if self.blend:  
            image_preds = self.image_output_head(image_features)
            clinical_preds = self.clinical_output_head(clinical_features)

            # and concatenate the outputs into a k+1 x N x C tensor where
            #   k is the number of modalities
            #   N is the batch size
            #   C is the number of classes (or targets)
            out = torch.stack((out, image_preds, clinical_preds), dim=0)
        
        # If were not doing gradient blending, than nothing else is needed just return logits of the multimodal output head as an N x C vector 
        return out
    
    @property
    def gradcam_layer(self):
        return self.image_model.model.backbone
    
    def add_gradcam(self, output_dir):
        '''
        If we want to use GradCAM with this model - we'll instantiate a related class with this object that includes the relevant functions & properties to use gradcam
        '''
        return MultiModalGradCAM(self)