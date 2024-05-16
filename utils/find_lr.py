import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import logging
import random
import numpy as np
from monai.transforms import \
    Compose, Resize, EnsureChannelFirst, ScaleIntensity, ToTensor, RandRotate, RandAxisFlip, RandZoom, RandGaussianNoise, \
    RandShiftIntensity, RandAdjustContrast, RandGaussianSmooth, RandGaussianSharpen, RandHistogramShift, RandRicianNoise
from monai.networks.nets import densenet121

from data.ImageDatasets import ImageDatasetByUIDs
from utils.utils import save_model, collate_fn, Normalize, criterion
from torch_lr_finder import LRFinder
from data.constants import IMAGE_DATA_MEAN, IMAGE_DATA_STDDEV

logger = logging.getLogger(__name__)


CLASSIFICATION_THRESHOLD = 0.5
NUM_CLASSES = 3

def find_lr(args, dataset):
    '''
    Tool for finding a decent starting point for the learning rate

    See papers by Leslie Smith, https://arxiv.org/abs/1506.01186
    '''
    # is_distributed = len(args.hosts) > 1 and args.backend is not None
    is_distributed = False
    logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = args.num_gpus > 0 and torch.cuda.is_available()
    logger.debug("Number of gpus available - {}".format(args.num_gpus))
    logger.debug('Cuda Available - {}'.format(torch.cuda.is_available()))
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    all_uids = dataset.uids
    random.seed(args.seed)
    random.shuffle(all_uids)
    train_uids = all_uids[:round(len(all_uids)* 0.8)]
    val_uids = all_uids[round(len(all_uids) *0.8):]

    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    train_transforms = Compose([
            EnsureChannelFirst(channel_dim=0),
            Normalize(IMAGE_DATA_MEAN, IMAGE_DATA_STDDEV),
            ScaleIntensity(),
            
            # Spatial Transformations
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandAxisFlip(prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            Resize(spatial_size=(64,64,64)),

            #Intensity Transformations
            RandShiftIntensity(0.1, prob=0.3),
            RandAdjustContrast(prob=0.3),
            RandGaussianSmooth(prob=0.2),
            RandGaussianSharpen(prob=0.2),
            RandHistogramShift(prob=0.3),
            RandGaussianNoise(prob=0.3, mean=0, std=0.05),
            
            ToTensor()
        ])

    val_transforms = Compose([
        EnsureChannelFirst(channel_dim=0),
        Normalize(IMAGE_DATA_MEAN, IMAGE_DATA_STDDEV),
        ScaleIntensity(),   
        Resize(spatial_size=(64,64,64)),
        ToTensor()
    ])
    
    with open('train_uids.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in train_uids]))
    with open('val_uids.txt', 'w') as f:
        f.write('\n'.join([str(x) for x in val_uids]))
    train_dataset = ImageDatasetByUIDs(dataset, train_uids, transforms=train_transforms)
    val_dataset = ImageDatasetByUIDs(dataset, val_uids, transforms=val_transforms)

    print("Training count =",len(train_dataset),"Validation count =", len(val_dataset))
            
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    # create model
    model = densenet121(
        spatial_dims=3,
        in_channels=1,
        out_channels=3
    ).to(device)

    loss_function = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), args.lr)
    
    optimizer = torch.optim.SGD(model.parameters(), 1e-7,
                                momentum=0.9,
                                nesterov=True,
                                weight_decay=1E-4)
    lr_finder = LRFinder(model, optimizer, loss_function, device=device)
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    lr_finder.plot() # to inspect the loss-learning rate graph
    lr_finder.reset() # to reset the model and optimizer to their initial state