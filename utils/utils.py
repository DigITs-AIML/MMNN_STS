import torch
import os
import logging
import torch.nn.functional as F
import torch.nn as nn
import torchvision
from typing import Dict, Iterable, Callable
from monai.transforms import Transform
from monai.utils.type_conversion import convert_to_tensor
import numpy as np
import boto3
import tempfile
import matplotlib.pyplot as plt
# from pycox.models.loss import Loss
from medcam import medcam
from data.utils import _stratifiedSplit

logger = logging.getLogger(__name__)

def criterion(loss_func, preds, labels, device):
  
  return loss_func(preds, labels).to(device)

def surv_criterion(loss_func, preds, events, durations, device):
    losses = 0
    
    for i in range(preds.shape[1]):
        losses += loss_func(preds[:, i], events[:, i], durations[:, i]).to(device)
    return losses        

def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

def collate_fn(data):
    images, targets = zip(*data)
    targets = torch.stack(targets)
    num_slices = []
    new_images = []
    for image in images:
        num_slices.append(image.shape[-1])

    min_slices = 32
    largest_image = max(num_slices) if max(num_slices) > min_slices else min_slices

    for image in images:
        slices = image.shape[-1]
        image = torch.Tensor(image)
        if slices < largest_image:
            padding = largest_image - slices
            # image = F.pad(image, (padding,0,0,0, 0, 0))

        new_images.append(image.float())
    
    new_images = torch.stack(new_images)
    # new_images = F.interpolate(new_images, (128,128,128))

    return new_images, targets

def collate_fn_surv(data):
    images, events, durations = zip(*data)
    events = torch.stack(events)
    durations = torch.stack(durations)
    num_slices = []
    new_images = []
    for image in images:
        num_slices.append(image.shape[-1])

    min_slices = 32
    largest_image = max(num_slices) if max(num_slices) > min_slices else min_slices

    for image in images:
        slices = image.shape[-1]
        image = torch.Tensor(image)
        if slices < largest_image:
            padding = largest_image - slices
            # image = F.pad(image, (padding,0,0,0, 0, 0))

        new_images.append(image.float())
    
    new_images = torch.stack(new_images)
    # new_images = F.interpolate(new_images, (128,128,128))

    return new_images, events, durations

def multimodal_collate_fn(data):

    mm_data, targets = zip(*data)
    new_data = {}
    images = [data['image'] for data in mm_data]
    clinical = [data['clinical'] for data in mm_data]
    new_data['image'] = torch.stack(images)
    new_data['clinical'] = torch.stack(clinical)
    targets = torch.stack(targets)

    new_data['image'] = new_data['image'].to(torch.float32)
    new_data['clinical'] = new_data['clinical'].to(torch.float32)

    new_data['image'] = convert_to_tensor(new_data['image'])

    return new_data, targets

def multimodal_collate_fn_surv(data):
    mm_data, events, durations = zip(*data)
    new_data = {}
    events = torch.stack(events)
    durations = torch.stack(durations)
    images = [data['image'] for data in mm_data]
    clinical = [data['clinical'] for data in mm_data]
    new_data['image'] = torch.stack(images)
    new_data['clinical'] = torch.stack(clinical)
    new_data['image'] = new_data['image'].to(torch.float32)
    new_data['clinical'] = new_data['clinical'].to(torch.float32)

    new_data['image'] = convert_to_tensor(new_data['image'])

    return new_data, events, durations


def stratifiedSplit(dataset, multimodal=False):
    if multimodal:
        # if a multimodal dataset was passed
        return _stratifiedSplit(dataset.clinical_dataset.data, dataset.clinical_dataset.targets, dataset.uids)
    else:
        try:
            # if a clinical dataset was passed
            return _stratifiedSplit(dataset.data, dataset.targets, dataset.uids)
        except:
            # if an image dataset was passed
            return _stratifiedSplit(dataset.classification_dataset.data, dataset.classification_dataset.targets, dataset.uids)


# Context manager for opening files from an S3 bucket
class S3Open():
    def __init__(self, URI):
        # Returns open file object pointing to file containing s3 data, treat is as any other open file object
        # It is the callers responsibility to close this object after it is no longer needed
        if URI.startswith('s3://'):
            logger.info('Detected S3 URI')

            # Parse URI for bucket and keys
            new_path = URI.replace('s3://', '')
            bucket_name = new_path.split(os.sep)[0]
            key_name = os.sep.join(new_path.split(os.sep)[1:])
            
            # Get the object
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(bucket_name)
            self.temp = None
            self.object = bucket.Object(key_name)
        else:
            raise ValueError('{} is not a valid S3 URI'.format(URI))
        
    def __enter__(self):
        # Create temporary file that will hold data from s3 object
        self.temp = tempfile.TemporaryFile()
        self.f = open(self.temp.name, 'rb+')
        
        # Download s3 object to local temporary file
        self.object.download_fileobj(self.f)
        
        # Return to start position of tempfile
        self.f.seek(0)

        return self.f
    
    def __exit__(self, exc_type, exc_value, tb):
        self.f.close()
        self.temp = None
        if exc_type is not None:
            return False
    
def loadUIDs(path):
    '''
    Loads uids stored in text file at specified path, one uid per line, returns a list of uids
    '''
    try:
        with open(path) as f:
            return [int(line.strip()) for line in f.readlines()]
    except Exception as e:
        if path.startswith('s3://'):
            logger.info('Detected S3 URI')

            # Parse URI for bucket and keys
            new_path = path.replace('s3://', '')
            bucket_name = new_path.split(os.sep)[0]
            key_name = os.sep.join(new_path.split(os.sep)[1:])
            
            # Get the object
            s3 = boto3.resource('s3')
            bucket = s3.Bucket(bucket_name)
            object = bucket.Object(key_name)
            
            # Create temporary file that will hold data from s3 object
            temp = tempfile.TemporaryFile()
            with open(temp.name, 'rb+') as f:
                # Download s3 object to local temporary file
                object.download_fileobj(f)
                
                # Return to start position of tempfile
                f.seek(0)

                # Same as for local file, readlines from temporary file, convert to integers to enable use of pandas
                return [int(line.strip()) for line in f.readlines()]
        else:
            logger.error('{} could not be found on local disk, nor is it a valid S3 URI'.format(path))
            raise(e)


    
class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, layers: Iterable[str], multimodal):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}

        if not multimodal:
            for layer_id in layers:
                layer = dict([*self.model.named_modules()])[layer_id]
                layer.register_forward_hook(self.save_outputs_hook(layer_id))
        else:
            self.model.output_head.features.register_forward_hook(self.save_outputs_hook('features'))

    def save_outputs_hook(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        _ = self.model(x)
        if len(self.layers) == 1:
            return self._features[self.layers[0]]
        return self._features

class BackpropagatableFeatureExtractor(nn.Module):
    # Enforces stricter model definitions compared to FeatureExtractor, but allows model to be trained through backpropagation
    # The reason this is needed is because the original feature extractor utilizes forward hooks, whose callbacks are only initiated after the models forward pass
    # Utilizing those callbacks inside of the forward pass is causing weirdness, so this class forgoes using hooks/callbacks at the expense of enforcing a 
    # specific structure for the model

    # Assumes model is broken into at least two sequential parts, 'backbone', and 'features'
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model.backbone(x)
        return self.model.features(x)

class MultiModalGradCAM(nn.Module):
    '''
    Custom GradCAM implementation for use with multimodal models - 3rd party solutions assume sequential CNN architectures.
    Generally, they do not support multimodal architectures or survival analysis models
    '''

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.layers = self.model.gradcam_layer.modules()
        self.input_shape = None
        
        # Quick algorithm to get the final convolutional layer in the network
        for layer in self.layers:
            if isinstance(layer, nn.Conv3d):
                self.layer = layer

        # Throw some hooks onto the final conv layer that will capture
        #   1. Activations during the forward pass - stored in self.features
        #   2. Gradients during the backward pass  - stored in self.grads
        self.layer.register_full_backward_hook(self.save_grads_hook())
        self.layer.register_forward_hook(self.save_outputs_hook())

    def save_outputs_hook(self):
        def fn(_, __, output):
            self.features = output
        return fn

    def save_grads_hook(self):
        def fn(_, __, grad_output):
            self.grads = grad_output[0]
        
        return fn

    def forward(self, x):
        outputs = self.model(x)
        self.input_shape = x['image'].shape
        att_maps = self.attentionMaps(outputs)
        return outputs, att_maps
    
    def attentionMaps(self, outputs):
        activations = self.features
        att_maps= []

        # Create one attention map for each class
        for cls in range(outputs.shape[1]):

            # We assume this function was only called after a forward pass and self.features is already populated
            # Run a backward pass so self.grads is populated - retain the graph since we need to run separate backward passes for each class
            outputs[0, cls].backward(retain_graph=True)

            # Capture gradients from the hook
            grads = self.grads

            # pooled gradient across channels
            pooled_grads = torch.mean(grads, dim=[0,2,3,4])

            # weight channel activations by pooled gradients
            num_channels = activations.shape[1]

            for i in range(num_channels):
                activations[:,i,...] *= pooled_grads[i]

            # average across channels
            heatmap = torch.mean(activations, dim=1).squeeze()

            heatmap_min = torch.min(heatmap)
            heatmap = heatmap - heatmap_min

            # normalize
            heatmap /= torch.max(heatmap)

            att_map = heatmap.squeeze()

            # if our attention map includes a batch dimension - this should really throw an error. We don't want to compute attention maps for multiple
            # input images at a single time
            # att_map should have 3 dimensions, one for each spatial dimension of the input image 
            # For testing though, we'll just select the first image in the batch
            # if att_map.ndim == 4:
            #     att_map = att_map[0,...]

            assert att_map.ndim == 3, 'Batch dimension found in attention map - Must use batch size 1 when computing attention maps'

            # temporarily add in Batch and Channel dimensions so that pytorch will know all of our current dimensions are spatial
            # Needed for interpolation, but they'll need to be squeezed out
            att_map = att_map[None, None, ...]
            att_map = F.interpolate(att_map, self.input_shape[2:], mode='trilinear').squeeze()
            

            att_maps.append(att_map)

        return att_maps



class Normalize(Transform):
    def __init__(self, mean, std):
        self.mean = mean
        self.stddev = std

    def __call__(self, image):
        # normalization formula is img = (img - mean * max_pixel_value) / (std * max_pixel_value)
        return (image - self.mean * np.max(image)) / (self.stddev * np.max(image))
    
def loadWeights(model, path, device):
    try:
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint)
        logger.info('Loaded provided weights from disk')

    except Exception as e1:
        if path.startswith('s3://'):
            with S3Open(path) as f:
                model.load_state_dict(torch.load(f, map_location=device))
            logger.info('Loaded provided weights from S3')
        elif path.endswith('DenseNet121_BHB-10K_yAwareContrastive.pth'):
            try:
                new_checkpoint = {}
                checkpoint = torch.load(path, map_location=device)

                # This block will modify the keys of the DenseNet121 pretrained model 
                # to align with keys of the monai implementation of densenet121 so things will load properly
                for key in checkpoint['model'].keys():
                    new_key = key.replace('module.', '')
                    heirarchy = new_key.split('.')
                    if heirarchy[0] == 'features' and heirarchy[1].startswith('dense'):
                        heirarchy.insert(3, 'layers')
                    new_key = '.'.join(heirarchy)
                    new_checkpoint[new_key] = checkpoint['model'][key]

                # We set strict = False, since the pretrained backbone does not include weights for layers in the classification head
                model.load_state_dict(new_checkpoint, strict=False)
                logger.info('Loaded pretrained backbone')
            except Exception as e2:
                raise e2
        else:
            raise e1
    return model

class LossTracker:

    '''
    Class used to find sources of loss. Useful in debugging convergence issues.
    
    '''
    def __init__(self):
        self.tp_loss = []
        self.fp_loss = []
        self.tn_loss = []
        self.fn_loss = []
        self.total_loss = []

        self.vs_loss = []
        self.dm_loss = []
        self.reop_loss = []

    def update(self, preds, labels, loss):
        # preds - NxC tensor containing thresholded predictions of network
        # labels - NxC tensor containing ground truth labels
        # loss - NxC tensor containing unreduced loss values (reduction of loss function must be set to 'none' - otherwise this function will throw errors)

        # Find index of tps, fps etc using thresholded predictions. Use those to index the corresponding loss tensor to separate the losses
        # For class losses, each class is one column in the loss tensor
         
        tps = (preds == 1) * (labels == 1)
        fps = (preds == 1) * (labels == 0)
        fns = (preds == 0) * (labels == 1)
        tns = (preds == 0) * (labels == 0)

        self.tp_loss.append(torch.mean(loss[tps]).item())
        self.fp_loss.append(torch.mean(loss[fps]).item())
        self.fn_loss.append(torch.mean(loss[fns]).item())
        self.tn_loss.append(torch.mean(loss[tns]).item())
        self.total_loss.append(torch.mean(loss).item())

        self.vs_loss.append(torch.mean(loss[:, 0]).item())
        self.dm_loss.append(torch.mean(loss[:, 1]).item())


    def save_plots(self):
        plt.plot(self.tp_loss, label='tp loss')
        plt.plot(self.fp_loss, label='fp loss')
        plt.plot(self.fn_loss, label='fn loss')
        plt.plot(self.tn_loss, label='tn loss')
        plt.plot(self.total_loss, label='all loss (mean)')
        # plt.plot(self.total_loss, label='total loss')
        plt.legend()
        plt.savefig('val_loss_by_cm.png')
        plt.clf()

        plt.plot(self.vs_loss, label='VS loss')
        plt.plot(self.dm_loss, label='DM loss')
        plt.plot(self.total_loss, label='All loss (mean)')
        # plt.plot(self.total_loss, label='Total loss')
        plt.legend()
        plt.savefig('val_loss_by_class.png')
        plt.clf()

def add_gradcam(model, output_dir='attention_maps', multimodal=False):
    if multimodal:
        return model.add_gradcam(output_dir)
    else:
        return medcam.inject(model, output_dir="attention_maps", save_maps=False, return_attention=True)