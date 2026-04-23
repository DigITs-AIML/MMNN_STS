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
        blend=False,
        pathology_model=None
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
        self.pathology_model = pathology_model
        self.clinical_predictors = clinical_predictors
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_clinical_inputs = len(clinical_predictors)
        self.clinical_model = MLP(self.num_clinical_inputs, self.num_classes, self.num_features)
        # self.output_head = MLP(self.num_features * 2, self.num_classes)
        if self.pathology_model is not None:
            self.output_head = nn.Linear(self.num_features * 3, self.num_classes)
        else:
            self.output_head = nn.Linear(self.num_features * 2, self.num_classes)
        self.blend=blend

        # Build clinical and image feature extractors
        # self.image_model = FeatureExtractor(self.image_model, ['features'])
        # self.clinical_model = FeatureExtractor(self.clinical_model, ['features'])

        self.image_model = BackpropagatableFeatureExtractor(self.image_model)
        self.clinical_model = BackpropagatableFeatureExtractor(self.clinical_model)
        if self.pathology_model is not None:
            self.pathology_model = BackpropagatableFeatureExtractor(self.pathology_model)

        self.clinical_output_head = nn.Linear(self.num_features, self.num_classes)
        self.image_output_head = nn.Linear(self.num_features, self.num_classes)
        self.pathology_output_head = nn.Linear(self.num_features, self.num_classes)



    def forward(self, x):
        print(x.keys())
        image_data = x['image']
        clinical_data = x['clinical']
        if self.pathology_model is not None:
            pathology_data = x['wsi']
            pathology_features = self.pathology_model(pathology_data)
            print('path model working')
        else:
            pathology_data = None

        image_features = self.image_model(image_data)
        clinical_features = self.clinical_model(clinical_data)

        if self.pathology_model is None:
            features = torch.cat([image_features, clinical_features], 1)
        else:
            features = torch.cat([image_features, clinical_features, pathology_features], 1)

        out = self.output_head(features)

        if self.blend:
            image_preds = self.image_output_head(image_features)
            clinical_preds = self.clinical_output_head(clinical_features)
            if self.pathology_model is not None:
                pathology_preds = self.pathology_output_head(pathology_features)
                out = torch.stack((out, image_preds, clinical_preds, pathology_preds), dim=0)
            else:
                out = torch.stack((out, image_preds, clinical_preds), dim=0)
        return out
    
    @property
    def gradcam_layer(self):
        return self.image_model.model.backbone
    
    def add_gradcam(self, output_dir):
        # self.image_model = medcam.inject(self.image_model, output_dir=output_dir, save_maps=True)
        # return self

        return MultiModalGradCAM(self)
    
    
    # def to(self, device):
    #     self.clinical_model = self.clinical_model.to(device)
    #     self.image_model = self.image_model.to(device)
    #     self.output_head = self.output_head.to(device)

    #     return self