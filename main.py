from data.constants import IMAGE_DATA_MEAN, IMAGE_DATA_STDDEV, NUM_DURATIONS, NUM_CLASSES, CLASS_FREQUENCIES
from losses.losses import CoxPH, focal_binary_cross_entropy, MultilabelBCELoss
from parser.parser import Parser
from data.ImageDatasets import ImageDatasetByUIDs

import argparse
import json
import logging
import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from PIL import Image
from monai.config import print_config
from monai.transforms import \
    Compose, Resize, EnsureChannelFirst, ScaleIntensity, ToTensor, RandRotate, RandAxisFlip, RandZoom, RandGaussianNoise, \
    RandShiftIntensity, RandAdjustContrast, RandGaussianSmooth, RandGaussianSharpen, RandHistogramShift, RandRicianNoise

from monai.metrics import compute_roc_auc, get_confusion_matrix, compute_confusion_matrix_metric
from monai.losses import FocalLoss
import random
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.utils import resample

from losses.GradientBlender import GradientBlender
from utils.find_lr import find_lr
from utils.utils import criterion, surv_criterion, save_model, collate_fn, collate_fn_surv, multimodal_collate_fn, multimodal_collate_fn_surv, loadUIDs, FeatureExtractor, Normalize, \
    stratifiedSplit, loadWeights, LossTracker, add_gradcam

import boto3

'''
This module serves as the main entry point to the training pipeline.

Four primary functions are implemented in this module

train_classification
train_survival

inference_classification
inference_survival

Which implement the training and evaluation loops for the MMNN.
'''


CLASSIFICATION_THRESHOLD = 0.5
NUM_SURV_FEATURES=50
SPATIAL_SIZE = (64,64,64)
NUM_BOOTSTRAP_ITERATIONS = 50
SUPER_BATCH_SIZE=64
BUCKET_NAME = 'bucket_name'
train_transforms = Compose([
            EnsureChannelFirst(channel_dim=0),
            Normalize(IMAGE_DATA_MEAN, IMAGE_DATA_STDDEV),
            ScaleIntensity(),
            
            # Spatial Transformations
            RandRotate(range_x=15, prob=0.5, keep_size=True),
            RandAxisFlip(prob=0.5),
            RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
            Resize(spatial_size=SPATIAL_SIZE),

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
        Resize(spatial_size=SPATIAL_SIZE),
        ToTensor()
    ])

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

def getF1Score(tps, fps, fns):
    f1s = []
    for idx in range(NUM_CLASSES):
        f1 = tps[idx] / (tps[idx] + 0.5 * (fns[idx] + fps[idx]))
        f1s.append(f1.item())
    
    return f1s

def getCIndices(preds, events, durations):
    '''
    args: preds, events, and durations are N x C tensors, where N is arbitary and C is the number of classes
        They must be aligned so that values index [i,:] corresponds to the same patient for all provided tensors

        preds: log hazard ratio predictions from the DL model
        events: Binary variable indicating whether event was censored to patient in question (0 - did not experience event, 1 - did experience event)
        durations: Time elapsed between start of follow up period and event for uncensored patients, for censored patients this is the full duration of the follow up period

    returns: List of length C containing Haral C-Indexes corresponding to each class

    The 'concordance index' function is from the lifelines package - 
        `from lifelines.utils import concordance_index`
    '''
    indices = []
    for i in range(NUM_CLASSES):
        indices.append(concordance_index(durations[:, i], preds[:, i], events[:,i]))
    return indices

def train_classification(model, train_dataset, val_dataset, args, device):

    use_cuda = model_args.num_gpus > 0 and torch.cuda.is_available()

    collater = None
    if args.multimodal:
        collater = multimodal_collate_fn
    else:
        collater = collate_fn
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater, num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater, num_workers=args.num_workers, pin_memory=use_cuda)

    # create model
    model = model.to(device)
    epoch_num = args.epochs
    val_interval = 1
    steps_per_epoch=0
    if len(train_dataset) % args.batch_size == 0:
        steps_per_epoch = int(len(train_dataset) / args.batch_size)
    else:
        steps_per_epoch = 1 + int(len(train_dataset) / args.batch_size)

    class_freqs = torch.tensor(CLASS_FREQUENCIES)
    pos_weights = (torch.ones_like(class_freqs) - class_freqs) / class_freqs
    pos_weights = pos_weights.to(device)

    # loss_function = torch.nn.CrossEntropyLoss()
    train_loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='sum') # sum of (y * log(p(y_hat) + (1-y) * log (1 - p(y_hat)))
    loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weights, reduction='none')
    # loss_function = FocalLoss(reduction='mean', gamma=2.0, include_background=True, weight=pos_weights)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=epoch_num)

    if args.blend:
        gradient_blender = GradientBlender(loss_function, device=device)

    #train model
    best_metric = -1
    best_metric_epoch = -1
    best_f1s = None
    epoch_train_loss_values = list()
    epoch_val_loss_values = list()
    metric_values = list()
    train_metrics_values = list()
    loss_tracker = LossTracker()
    for epoch in range(epoch_num):
        logger.info('-' * 10)
        logger.info(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        tps = 0
        fps = 0
        fns = 0

        if args.blend:
            train_preds = torch.tensor([], dtype=torch.float32, device='cpu')
            train_gt = torch.tensor([], dtype=torch.long, device='cpu')
            val_preds = torch.tensor([], dtype=torch.float32, device='cpu')
            val_gt = torch.tensor([], dtype=torch.long, device='cpu')


        for batch_data in train_loader:

            # Set input_validation = 1 to output cropped volumes for manual verification in 3D slicer / OHIF
            step += 1
            inputs, labels = None, None
            #inputs is dimensions BCDHW, targets is dimensions BxNumClass
            if not args.multimodal:
                inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            else:
                inputs = batch_data[0]
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(device)
                labels = batch_data[1].to(device)

            optimizer.zero_grad()
            # outputs = torch.sigmoid(model(inputs))
            outputs = model(inputs)
            if args.blend:
                loss = gradient_blender.computeLoss(outputs, labels)
            else:
                loss = criterion(train_loss_function, outputs, labels, device)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            logger.info(f"{step}/{1+len(train_loader.dataset) // train_loader.batch_size}, train_loss: {loss.item():.4f}")
            epoch_len = len(train_loader.dataset) // train_loader.batch_size
            scheduler.step()
            
            outputs = torch.sigmoid(outputs)

            if args.blend:
                train_preds = torch.cat([train_preds, outputs.detach().cpu()], dim=1)
                train_gt = torch.cat([train_gt, labels.detach().cpu()], dim=0)
                outputs = outputs[0, ...]

            preds = outputs > CLASSIFICATION_THRESHOLD
            tps += torch.sum((preds == 1) * (labels == 1), 0)
            fps += torch.sum((preds == 1) * (labels == 0), 0)
            fns += torch.sum((preds == 0) * (labels == 1), 0)

        f1s = getF1Score(tps, fps, fns)
        train_metrics_values.append(np.mean(f1s))


        epoch_loss = epoch_loss / len(train_dataset)
        epoch_train_loss_values.append(epoch_loss)
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y = torch.tensor([], dtype=torch.long, device=device)
                losses = torch.tensor([], dtype=torch.float32, device=device)
                test_loss = 0
                for val_data in val_loader:
                    val_images, val_labels = None, None
                    #inputs is dimensions BCDHW, targets is dimensions BxNumClass
                    if not args.multimodal:
                        val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
                    else:
                        val_images = val_data[0]
                        for key in val_images.keys():
                            val_images[key] = val_images[key].to(device)
                        val_labels = val_data[1].to(device)
                    # preds = torch.sigmoid(model(val_images))
                    preds = model(val_images)
                    # test_loss += criterion(loss_function, preds, val_labels, device).item()
                    if args.blend:
                        loss = gradient_blender.computeLoss(preds,val_labels, no_reduce=True)
                    else:
                        loss = criterion(loss_function, preds, val_labels, device)
                    test_loss += torch.sum(loss).item()
                    preds = torch.sigmoid(preds)
                    preds = preds > CLASSIFICATION_THRESHOLD

                    if args.blend:
                        val_preds = torch.cat([val_preds, preds.detach().cpu()], dim=1)
                        val_gt = torch.cat([val_gt, val_labels.detach().cpu()], dim=0)

                        losses = torch.cat([losses, loss[0, ...]], dim=0)
                        y_pred = torch.cat([y_pred, preds[0, ...]], dim=0)
                        y = torch.cat([y, val_labels], dim=0)
                    
                    else:
                        losses = torch.cat([losses, loss], dim=0)
                        y_pred = torch.cat([y_pred, preds], dim=0)
                        y = torch.cat([y, val_labels], dim=0)

                loss_tracker.update(y_pred, y, losses)
                epoch_val_loss_values.append(test_loss / len(val_dataset))
                logger.info('Validation loss: {}'.format(test_loss / len(val_dataset)))
                
                tps = (y_pred == 1) * (y == 1)
                fps = (y_pred == 1) * (y == 0)
                fns = (y_pred == 0) * (y == 1)

                tps = torch.sum(tps, 0)
                fps = torch.sum(fps, 0) 
                fns = torch.sum(fns, 0)

                f1s = getF1Score(tps, fps, fns)
                f1s = np.array(f1s)
                mean_f1 = np.mean(f1s)
                metric_values.append(mean_f1)
                if mean_f1 > best_metric:
                    best_metric = mean_f1
                    best_f1s = f1s
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), './model.pth')
                    logger.info('saved new best metric model')
                logger.info(f"current epoch: {epoch + 1} current f1: {mean_f1:.4f}"
                      f" best f1: {best_metric:.4f} at epoch: {best_metric_epoch}")

        print(args.blend, (epoch+1) % args.blend_update_interval)
        if args.blend and (epoch + 1) % args.blend_update_interval == 0:
            print('Updating gradient blender weights')
            gradient_blender.updateWeights(train_preds, train_gt, val_preds, val_gt)
            print('Completed updating gradient blender weights - new weights : {}'.format(gradient_blender.weights))
    loss_tracker.save_plots()

    logger.info(f"train completed, best_metric: {best_metric:.4f}, best_f1s: {best_f1s} at epoch: {best_metric_epoch}")
    plt.plot(epoch_train_loss_values, label= 'Train Loss')

    torch.save(model.state_dict(), './final_model.pth')

    plt.plot(epoch_val_loss_values, label= 'Validation Loss')
    plt.legend()
    plt.savefig(os.path.join('.', 'train_val_loss.png'))

    plt.clf()
    # train_metrics_values = [x * np.max(epoch_train_loss_values) for x in train_metrics_values]
    # epoch_train_loss_values = [x *np.max(epoch_train_loss_values) for x in epoch_train_loss_values]
    plt.plot(train_metrics_values, label = 'Train F1 Score')
    plt.plot(metric_values, label='Validation F1 Score')
    plt.legend()
    plt.savefig(os.path.join('.', 'train_val_f1.png'))


def train(args, model, dataset, device):
    
    '''
    This function will handle some boilerplate common to classification and survival models, load uids for train and val sets, initialize train & val datasets
    It will then delegate training to the relevent train loop depending on if the task is survival analysis or classification
    '''
    is_distributed = False
    logger.debug("Distributed training - {}".format(is_distributed))

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    all_uids = dataset.uids
    random.seed(args.seed)
    random.shuffle(all_uids)

    if args.split:
        train_uids, val_uids, test_uids = stratifiedSplit(dataset, multimodal=args.multimodal)
    else:
        train_uids = loadUIDs(args.train_uid_location)
        val_uids = loadUIDs(args.val_uid_location)

    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    
    train_dataset = ImageDatasetByUIDs(dataset, train_uids, transforms=train_transforms)
    val_dataset = ImageDatasetByUIDs(dataset, val_uids, transforms=val_transforms)

    print("Training count =",len(train_dataset),"Validation count =", len(val_dataset))
    if args.survival:
        train_survival(model, train_dataset, val_dataset, args, device)
    else:
        train_classification(model, train_dataset, val_dataset, args, device)
    

    try:
        s3 = boto3.resource('s3')
        if args.survival:
            s3.Bucket(BUCKET_NAME).upload_file('./train_val_loss.png', '{}/train_val_loss.png'.format(args.output_dir))
            s3.Bucket(BUCKET_NAME).upload_file('./train_val_c_score.png', '{}/train_val_f1.png'.format(args.output_dir))
            s3.Bucket(BUCKET_NAME).upload_file('./best_surv_model.pth', '{}/model.pth'.format(args.output_dir))
            if args.blend:
                s3.Bucket(BUCKET_NAME).upload_file('./gblend_weights_history.csv', '{}/gblend_weights_history.csv'.format(args.output_dir))
        else:
            s3.Bucket(BUCKET_NAME).upload_file('./train_val_loss.png', '{}/train_val_loss.png'.format(args.output_dir))
            s3.Bucket(BUCKET_NAME).upload_file('./train_val_f1.png', '{}/train_val_f1.png'.format(args.output_dir))
            s3.Bucket(BUCKET_NAME).upload_file('./model.pth', '{}/model.pth'.format(args.output_dir))
            s3.Bucket(BUCKET_NAME).upload_file('./final_model.pth', '{}/final_model.pth'.format(args.output_dir))
            s3.Bucket(BUCKET_NAME).upload_file('./val_loss_by_class.png', '{}/val_loss_by_class.png'.format(args.output_dir))
            s3.Bucket(BUCKET_NAME).upload_file('./val_loss_by_cm.png', '{}/val_loss_by_cm.png'.format(args.output_dir))
    except Exception as e:
        print('Could not save to s3 bucket - no connection to S3')
        print(e)
        pass

def train_survival(model, train_dataset, val_dataset, args, device):

    use_cuda = args.num_gpus > 0 and torch.cuda.is_available()
    collater = None
    if args.multimodal:
        collater = multimodal_collate_fn_surv
    else:
        collater = collate_fn_surv
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater, num_workers=args.num_workers, pin_memory=use_cuda)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collater, num_workers=args.num_workers, pin_memory=use_cuda)

    # create model
    model = model.to(device)
    epoch_num = args.epochs
    val_interval = 1
    steps_per_epoch=0

    # defining constants relevent to gradient accumulation and LR scheduling
    super_batch_interval = SUPER_BATCH_SIZE /args.batch_size
    if len(train_dataset) % SUPER_BATCH_SIZE == 0:
        steps_per_epoch = int(len(train_dataset) / SUPER_BATCH_SIZE)
    else:
        steps_per_epoch = 1 + int(len(train_dataset) / SUPER_BATCH_SIZE)

    loss_function = CoxPH
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=epoch_num)

    if args.blend:
        gradient_blender = GradientBlender(loss_function, survival=True, surv_criterion=surv_criterion)

     #train model
    best_loss = np.inf
    best_loss_epoch = -1
    best_c_indexes = None
    epoch_train_loss_values = list()
    epoch_val_loss_values = list()
    train_metric_values = list()
    val_metric_values = list()
    for epoch in range(epoch_num):
        logger.info('-' * 10)
        logger.info(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        
        # Define some placeholders that will hold our train & val preds & labels
        # When using gradient blending, they will be used to calculate gblend weight updates
        # When not using gradient blending - these aren't strictly necessary but we're using them to track train&val loss over the course of training
        with torch.no_grad():
            c_pred = torch.tensor([], dtype=torch.float32, device=device)
            c_events = torch.tensor([], dtype=torch.long, device=device)
            c_durations = torch.tensor([], dtype=torch.long, device=device)
            y_pred = torch.tensor([], dtype=torch.float32, device=device)
            y_events = torch.tensor([], dtype=torch.long, device=device)
            y_durations = torch.tensor([], dtype=torch.long, device=device)

        for i, batch_data in enumerate(train_loader):

            step += 1

            # Sends data to GPU
            if not args.multimodal:
                inputs, events, durations = batch_data[0].to(device), batch_data[1].to(device), batch_data[2].to(device)
            else:
                inputs = batch_data[0]
                for key in inputs.keys():
                    inputs[key] = inputs[key].to(device)
                events = batch_data[1].to(device)
                durations = batch_data[2].to(device)
            
            # Forward pass
            outputs = model(inputs)

            # Compute loss
            if args.blend:
                loss, _ = gradient_blender.computeLoss(outputs, events, durations)
            else:
                loss = surv_criterion(loss_function, outputs, events, durations, device)

            # Backward pass
            loss.backward()

            # Logging
            epoch_loss += loss.item()
            batch_loss = loss.item() / args.batch_size
            logger.info(f"{step}/{1+len(train_loader.dataset) // train_loader.batch_size}, train_loss: {batch_loss:.4f}")

            # gradient accumulation prior to stepping optimizer
            # step on the super batch interval and on the last batch of an epoch
            if (i+1) % super_batch_interval == 0 or i == len(train_loader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Log train preds & labels
            if args.blend:
                c_pred = torch.cat([c_pred, outputs.detach()], dim=1)
                c_events = torch.cat([c_events, events.detach()], dim=0)
                c_durations = torch.cat([c_durations, durations.detach()], dim=0)
                
            else:
                c_pred = torch.cat([c_pred, outputs.detach()], dim=0)
                c_events = torch.cat([c_events, events.detach()], dim=0)
                c_durations = torch.cat([c_durations, durations.detach()], dim=0)
        
        # Compute c-index on train set
        if args.blend:
            train_c_index = getCIndices(c_pred[0,...].cpu(), c_events.cpu(), c_durations.cpu())
        else:
            train_c_index = getCIndices(c_pred.cpu(), c_events.cpu(), c_durations.cpu())
        
        # store c-index for logging purposes
        train_metric_values.append(np.mean(train_c_index))

        epoch_loss /= len(train_dataset)
        epoch_train_loss_values.append(epoch_loss)
        logger.info(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        # Validation
        if (epoch + 1) % val_interval == 0:
            model.eval()

            with torch.no_grad():

                # Clear tensors holding previous validation data
                y_pred = torch.tensor([], dtype=torch.float32, device=device)
                y_events = torch.tensor([], dtype=torch.long, device=device)
                y_durations = torch.tensor([], dtype=torch.long, device=device)
                test_step = 0
                test_loss = 0
                for val_data in val_loader:
                    test_step+=1

                    # Send data to gpu
                    if not args.multimodal:
                        val_images, val_events, val_durations = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
                    else:
                        val_images = val_data[0]
                        for key in inputs.keys():
                            val_images[key] = val_images[key].to(device)
                        val_events = val_data[1].to(device)
                        val_durations = val_data[2].to(device)

                    # forward pass
                    preds = model(val_images)

                    # loss computation
                    if args.blend:
                        loss, selection_loss = gradient_blender.computeLoss(preds, val_events, val_durations)
                    else:
                        loss = surv_criterion(loss_function, preds, val_events, val_durations, device).item()
                        selection_loss = loss

                    test_loss += loss

                    # Cat preds along the batch dimension
                    if args.blend:
                        y_pred = torch.cat([y_pred, preds], dim=1)
                    else:
                        y_pred = torch.cat([y_pred, preds], dim=0)
                    y_events = torch.cat([y_events, val_events], dim=0)
                    y_durations = torch.cat([y_durations, val_durations], dim=0)

                # Move these to cpu so they aren't eating up VRAM
                y_pred = y_pred.cpu()
                y_events = y_events.cpu()
                y_durations = y_durations.cpu()

                # Compute c-indices
                if args.blend:
                    # If using gblend - we only care about the cindex of predictions made by the multimodal output head, which always has index 0 in the first dimension
                    c_indices = getCIndices(y_pred[0, ...].numpy(), y_events.numpy(), y_durations.numpy())
                else:
                    c_indices = getCIndices(y_pred.numpy(), y_events.numpy(), y_durations.numpy())

                mean_c_index = np.mean(c_indices)
                val_metric_values.append(mean_c_index)
                print('All C-indexes: {}'.format(c_indices))
                print('Mean C-index: {}'.format(mean_c_index))

                epoch_val_loss_values.append(test_loss/len(val_dataset))
                print('validation loss: {}'.format(test_loss/len(val_dataset)))

                # The selection loss is the unweighted component of the multimodal output head loss
                if selection_loss < best_loss:
                    best_loss = selection_loss
                    best_loss_epoch = epoch + 1
                    best_c_indexes = c_indices
                    torch.save(model.state_dict(), 'best_surv_model.pth')
                    logger.info('saved new best metric model')
                logger.info(f"current epoch: {epoch + 1} current loss: {selection_loss:.4f}"
                      f" best loss: {best_loss:.4f} at epoch: {best_loss_epoch}"
                      f" best c_indices: {best_c_indexes} with mean: {np.mean(best_c_indexes):.3f}")
                
        # Update gradient blender weights
        if args.blend and (epoch + 1) % args.blend_update_interval == 0:
            print('Updating gradient blender weights')

            gradient_blender.updateWeights(c_pred, c_events, c_durations, y_pred, y_events, y_durations)
            print('Completed updating gradient blender weights - new weights : {}'.format(gradient_blender.weights))
    if args.blend:
        gradient_blender.saveHistory()            
    plt.plot(epoch_train_loss_values, label= 'Train Loss')

    plt.plot(epoch_val_loss_values, label= 'Validation Loss')
    plt.legend()
    plt.savefig(os.path.join('.', 'train_val_loss.png'))

    plt.clf()
    plt.plot(train_metric_values, label='Train C Score')
    plt.plot(val_metric_values, label='Validation C Score')
    plt.legend()
    plt.savefig(os.path.join('.', 'train_val_c_score.png'))



def inference(args, model, dataset, device, save_probs=True):

    model.eval()
    # send model to device
    model = model.to(device)

    # load validation uids from txt
    uids  = loadUIDs(args.test_uid_location)

    collater = None
    if args.multimodal:
        collater = multimodal_collate_fn
    else:
        collater = collate_fn

    # create dataloader from loaded uids
    val_dataset = ImageDatasetByUIDs(dataset, uids, transforms=val_transforms)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collater, num_workers=args.num_workers)
    
    feature_group_for_extraction = 'features' #'backbone' for monai densenet, 'features' for custom num features
    feature_extractor = FeatureExtractor(model, [feature_group_for_extraction], args.multimodal)
    [print(module) for module in model.modules() if not isinstance(module, nn.Sequential)]

    # with torch.no_grad(): - need gradients enabled for gradcam to work
    y_pred = torch.tensor([], dtype=torch.float32, device=device)
    y = torch.tensor([], dtype=torch.float32, device=device)
    y_probs = torch.tensor([], dtype=torch.float32, device=device)
    all_flattened_features = torch.tensor([], dtype=torch.float32, device=device)
    all_reduced_features = torch.tensor([], dtype=torch.float32, device=device)

    # inference loop
    for i, val_data in enumerate(val_loader):
        val_images, val_labels = None, None
        #inputs is dimensions BCDHW, targets is dimensions BxNumClass
        if not args.multimodal:
            val_images, val_labels = val_data[0].to(device), val_data[1].to(device)
        else:
            val_images = val_data[0]
            for key in val_images.keys():
                val_images[key] = val_images[key].to(device)
            val_labels = val_data[1].to(device)


        features = feature_extractor(val_images)

        uid = uids[i]
        print(uid)

        if args.no_gradcam:
            preds = model(val_images)
        else:
            preds, att_maps = model(val_images)
        probs = torch.sigmoid(preds)
        preds = probs > CLASSIFICATION_THRESHOLD
        y_probs = torch.cat([y_probs, probs], dim=0)
        y_pred = torch.cat([y_pred, preds], dim=0)
        y = torch.cat([y, val_labels], dim=0)

        # This section will handle attention maps
        if not args.no_gradcam:
            if not os.path.exists('./attention_maps'):
                os.makedirs('./attention_maps')

            for j in range(preds.shape[0]):

                correctness = preds == val_labels

                t1_img = val_images[j, 0, :].numpy()
                t2_img = val_images[j, 1, :].numpy()
                att_map = np.squeeze(att_maps[j, :].numpy())
                output_string = []

                for ctr in range(preds.shape[1]):
                    if correctness[j, ctr].item() == True:
                        output_string.append('1')
                    else:
                        output_string.append('0')
                output_string.append('_')
                for ctr in range(preds.shape[1]):
                    if val_labels[j, ctr].item() == 1:
                        output_string.append('1')
                    else:
                        output_string.append('0')
                output_string.append('_')
                for ctr in range(preds.shape[1]):
                    if preds[j, ctr].item() == 1:
                        output_string.append('1')
                    else:
                        output_string.append('0')
                output_string.append('_')
                        
                output_string = ''.join(output_string)

                # print(t1_img.shape, t2_img.shape, att_map.shape)

                patient_path = '.{}{}{}{}'.format(os.sep, 'attention_maps', os.sep, '{}_patient_{}'.format(output_string, uid))
                if os.path.exists(patient_path):
                    for filename in os.listdir(patient_path):
                        os.remove(os.path.join(patient_path, filename))
                    os.removedirs(patient_path)

                os.mkdir(patient_path)

                t1_img = nib.Nifti1Image(t1_img, affine=np.eye(4))
                t2_img = nib.Nifti1Image(t2_img, affine=np.eye(4))
                att_map = nib.Nifti1Image(att_map, affine=np.eye(4))
                nib.save(t1_img, os.path.join(patient_path, 't1image.nii.gz'))
                nib.save(t2_img, os.path.join(patient_path, 't2image.nii.gz'))
                nib.save(att_map, os.path.join(patient_path, 'att_map.nii.gz'))
                with open(os.path.join(patient_path, 'probabilities.txt'), 'w') as f:
                    lines = [str(prob.item()) for prob in probs[j, ...]]
                    [f.write(line) for line in lines]

        extracted_features = features
        flattened_features = torch.flatten(extracted_features)[None, ...]
        reduced_features = torch.flatten(torch.mean(extracted_features, dim=1))[None, ...]
        all_flattened_features = torch.cat([all_flattened_features, flattened_features], dim=0)
        all_reduced_features = torch.cat([all_reduced_features, reduced_features], dim=0)

    tps = (y_pred == 1) * (y == 1)
    fps = (y_pred == 1) * (y == 0)
    fns = (y_pred == 0) * (y == 1)

    num_samples = tps.shape[0]
    tps = torch.sum(tps, 0)
    fps = torch.sum(fps, 0) 
    fns = torch.sum(fns, 0)

    f1s = []
    for idx in range(NUM_CLASSES):
        f1 = tps[idx] / (tps[idx] + 0.5 * (fns[idx] + fps[idx]))
        f1s.append(f1.item())
    print(f1s)
    print(np.mean(f1s))
    if save_probs:
        y_probs = y_probs.cpu().detach().numpy()

        # Create a copy to not interfere with save_features function
        y_copy = y.cpu().detach().numpy()

        probs_and_labels = np.concatenate((np.array(uids)[..., None].astype(int), y_probs, y_copy), axis=1)
        df = pd.DataFrame(probs_and_labels)
        df.to_csv('model_probabilities.csv', index=False)


def inference_survival(args, model, dataset, device, save_preds=True):

    model.eval()
    # send model to device
    model = model.to(device)

    # load validation uids from txt
    uids  = loadUIDs(args.train_uid_location)

    collater = None
    if args.multimodal:
        collater = multimodal_collate_fn_surv
    else:
        collater = collate_fn_surv

    all_c_indices = []

    if args.bootstrap:
        all_uids = [resample(uids) for _ in range(NUM_BOOTSTRAP_ITERATIONS)] # from sklearn.utils import resample
    else:
        all_uids = [uids]

    for k in range(len(all_uids)):

        if args.bootstrap:
            print('Bootstrap Iteration: {}'.format(k+1))
            args.no_gradcam = True
            save_preds = False
        
        uids = all_uids[k]
        # create dataloader from loaded uids
        val_dataset = ImageDatasetByUIDs(dataset, uids, transforms=val_transforms)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collater)#, num_workers=args.num_workers, pin_memory=True)
        
        # with torch.no_grad(): - need gradients enabled for gradcam to work
        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y_events = torch.tensor([], dtype=torch.float32, device=device)
        y_durations = torch.tensor([], dtype=torch.float32, device=device)

        # inference loop
        for i, val_data in enumerate(val_loader):
            val_images, val_events, val_durations = None, None, None
            #inputs is dimensions BCDHW, targets is dimensions BxNumClass
            if not args.multimodal:
                val_images, val_events, val_durations = val_data[0].to(device), val_data[1].to(device), val_data[2].to(device)
            else:
                val_images = val_data[0]
                for key in val_images.keys():
                    val_images[key] = val_images[key].to(device)
                val_events = val_data[1].to(device)
                val_durations = val_data[2].to(device)

            uid = uids[i]
            if not args.bootstrap:
                print(uid)

            if args.no_gradcam:
                preds = model(val_images)
            else:
                preds, att_maps = model(val_images)

            y_events = torch.cat([y_events, val_events], dim=0)
            y_pred = torch.cat([y_pred, preds], dim=0)
            y_durations = torch.cat([y_durations, val_durations], dim=0)

            # This section will handle attention maps
            if not args.no_gradcam:
                if not os.path.exists('./attention_maps'):
                    os.makedirs('./attention_maps')

                for j in range(preds.shape[0]):

                    t1_img = val_images['image'][j, 0, :].numpy()
                    t2_img = val_images['image'][j, 1, :].numpy()
                    if not args.multimodal:
                        att_map = np.squeeze(att_maps[j, :].numpy())
                    else:
                        att_map = att_maps[0].cpu().detach().numpy() # This selects the vitalstatus attention map , ideally we'd save each seperately

                    patient_path = '.{}{}{}{}'.format(os.sep, 'attention_maps', os.sep, '_patient_{}'.format(uid))
                    if os.path.exists(patient_path):
                        for filename in os.listdir(patient_path):
                            os.remove(os.path.join(patient_path, filename))
                        os.removedirs(patient_path)

                    os.mkdir(patient_path)

                    t1_img = nib.Nifti1Image(t1_img, affine=np.eye(4))
                    t2_img = nib.Nifti1Image(t2_img, affine=np.eye(4))
                    att_map = nib.Nifti1Image(att_map, affine=np.eye(4))
                    nib.save(t1_img, os.path.join(patient_path, 't1image.nii.gz'))
                    nib.save(t2_img, os.path.join(patient_path, 't2image.nii.gz'))
                    nib.save(att_map, os.path.join(patient_path, 'att_map.nii.gz'))
                    with open(os.path.join(patient_path, 'preds.txt'), 'w') as f:
                        lines = [str(pred.item()) for pred in preds[j, ...]]
                        [f.write(line) for line in lines]
        
        y_pred = y_pred.cpu().detach().numpy()
        y_events = y_events.cpu().detach().numpy()
        y_durations = y_durations.cpu().detach().numpy()
        
        try:

            c_indices = getCIndices(y_pred, y_events, y_durations)
            mean_c = np.mean(c_indices)

        except ZeroDivisionError as e:
            if args.bootstrap:
                continue
            else:
                raise(e)
            
        if args.bootstrap:
            all_c_indices.append(c_indices)
        else:
            print('Inference complete - C-Indices: {}'.format(c_indices))
            print('Mean C-Index: {}'.format(mean_c))
            if save_preds:

                preds_and_labels = np.concatenate((np.array(uids)[..., None].astype(int), y_pred, y_events, y_durations), axis=1)
                cls_headers = ['VS', 'DM']
                pred_headers = [x+'_pred' for x in cls_headers]
                event_headers = [x+'_event' for x in cls_headers]
                duration_headers = [x+'_duration' for x in cls_headers]

                headers = ['MRN'] + pred_headers + event_headers + duration_headers
                df = pd.DataFrame(preds_and_labels, columns=headers)
                df.to_csv('survival_model_predictions.csv', index=False)
                s3 = boto3.resource('s3')
                s3.Bucket(BUCKET_NAME).upload_file('./survival_model_predictions.csv', '{}/survival_model_predictions.csv'.format(args.output_dir))
                print('uploaded to {}/survival_model_predictions.csv'.format(args.output_dir))
                
    if args.bootstrap:
        c_indices = np.array(all_c_indices)
        means = np.mean(c_indices, axis=0)
        stds = np.std(c_indices, axis=0)
        print('Mean c indices: {}'.format(means))
        print('Std. devs: {}'.format(stds))

def str_to_bool(arg):
    if arg.lower() == 'false':
        return False
    elif arg.lower() == 'true':
        return True
    else:
        raise ValueError('Unexpected value for boolean conversion: {}'.format(arg))
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--preop', action='store_true', help='Use dataset consisting of clinical features available preoperation')
    argparser.add_argument('--postop', action='store_true', help='Use dataset consisting of clinical features available pre and post operation')
    argparser.add_argument('--radiomics', action='store_true', help='Use dataset consisting of extracted radiomic features')
    argparser.add_argument('--images', action='store_true', help='Use dataset consisting of image data')
    argparser.add_argument('--classification', action='store_true', help='Binary classificiation for Survival & DM')
    argparser.add_argument('--survival', action='store_true', help='Time to Event model for survival & DM')
    argparser.add_argument('--segmentation', action='store_true', help='Perform tumor segmentation')
    argparser.add_argument('--lr_finder', action='store_true', help='Run LR finder to find good initial learning rate see (Smith 2015) Cyclical Learning Rates for Training Neural Networks')
    argparser.add_argument('--no_gradcam', action='store_true', help='Disable gradcam for inference')

    # String equivalent booleans for sagemaker since sagemaker doesn't support action args
    argparser.add_argument('--use_images', type=str, default='false', help='Same as --images, for use with sagemaker')
    argparser.add_argument('--use_preop', type=str, default='false', help='Same as --preop, for use with sagemaker')
    argparser.add_argument('--use_postop', type=str, default='false', help='same as --postop, for use with sagemaker')
    argparser.add_argument('--classification_task', type=str, default='false', help='Same as --classification, for use with sagemaker')
    argparser.add_argument('--inference_task', type=str, default='false', help='Same as --inference, for use with sagemaker')
    argparser.add_argument('--survival_task', type=str, default='false', help='Same as --survival, for use with sagemaker')
    argparser.add_argument('--use_blend', type=str, default='false', help='Use gradient blending in multimodal models')
    
    argparser.add_argument('--weights', type=str, default='./weights/DenseNet121_BHB-10K_yAwareContrastive.pth', help='Path to pretrained weights')
    argparser.add_argument('--output_path', type=str, default='.', help='Directory for storing outputs')

    
    argparser.add_argument('--inference', action='store_true', help='Inference using validation set')

    argparser.add_argument('--data_loc', type=str, help='path to clinical data')
    argparser.add_argument('--image_loc', type=str, help='path to image data')
    argparser.add_argument('--key_loc', type=str, help='path to patient key')
    argparser.add_argument('--rad_loc', type=str, help='path to radiomic features')

    argparser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    argparser.add_argument('--lr', type=float, default=5E-4, help='Learning rate')
    argparser.add_argument('--split', action='store_true', help='Use this flag to create a new dataset split, otherwise will load existing')
    argparser.add_argument('--train_uid_location', type=str, default='./stratified_train_uids.txt', help='location of list of train uids')
    argparser.add_argument('--val_uid_location', type=str, default='./stratified_val_uids.txt', help='location of list of val uids')
    argparser.add_argument('--config', type=str, default='./config.yaml', help='Path to YAML configuration file (see example)')
    argparser.add_argument('--blend', action='store_true', help='Use Gradient blending method during training, only applicable for multimodal model')
    argparser.add_argument('--blend_update_interval', type=int, default=5, help='Number of epochs before updating gradient blending weights')
    argparser.add_argument('--bootstrap', action='store_true', help='Bootstrap evaluation')
    

    args = argparser.parse_args()
    args.images = args.images or str_to_bool(args.use_images)
    args.classification = args.classification or str_to_bool(args.classification_task)
    args.inference = args.inference or str_to_bool(args.inference_task)
    args.survival = args.survival or str_to_bool(args.survival_task)
    args.preop = args.preop or str_to_bool(args.use_preop)
    args.postop = args.postop or str_to_bool(args.use_postop)
    args.blend = args.blend or str_to_bool(args.use_blend)

    assert not all([args.classification, args.survival, args.segmentation]), 'Can only specify one of --classification , --survival , or --segmentation'
    assert any([args.classification, args.survival, args.segmentation]), 'Must specify one of --classification , --survival , or --segmentation'
    assert any([args.train, args.test, args.inference]), 'Must specify atleast one of --train, --test, --inference'

    configparser = Parser(args.config)
    config = configparser.parseConfig()

    image_path = None
    if args.images:
        image_path = configparser.getImagePath()

    dataset = configparser.getDatasets(args, image_path=image_path)
    model = configparser.getModel(args)

    multimodal = args.images and (args.preop or args.postop)

    blend = args.blend and multimodal
    model_args = {
        'batch_size': config['Hyperparameters']['train_batch_size'],
        'test_batch_size': config['Hyperparameters']['test_batch_size'],
        'epochs': config['Hyperparameters']['epochs'],
        'lr': config['Hyperparameters']['learning_rate'],
        'momentum': config['Hyperparameters']['momentum'],
        'weight_decay': config['Hyperparameters']['weight_decay'],
        'seed': config['Hyperparameters']['seed'],
        'log_interval': config['Hyperparameters'],
        'backend' : None,
        'num_gpus': config['Hyperparameters']['num_gpus'],
        'output_dir':config['Preprocessing']['output_dir'],
        'split': args.split, 
        'val_uid_location': config['Preprocessing']['val_uid_location'],
        'train_uid_location': config['Preprocessing']['train_uid_location'],
        'test_uid_location': config['Preprocessing']['test_uid_location'],
        'survival' : args.survival,
        'num_workers' : config['Preprocessing']['num_workers'],
        'multimodal' : multimodal,
        'blend': blend,
        'blend_update_interval' : args.blend_update_interval,
        'no_gradcam': args.no_gradcam,
        'bootstrap': args.bootstrap
    }

    if multimodal:
        model.blend = blend
    model_args = argparse.Namespace(**model_args)

    use_cuda = model_args.num_gpus > 0 and torch.cuda.is_available()
    logger.debug("Number of gpus available - {}".format(model_args.num_gpus))
    logger.debug('Cuda Available - {}'.format(torch.cuda.is_available()))
    kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.inference:
        model = loadWeights(model, config['Hyperparameters']['model_weights'], device)
    else:
        try:
            model = loadWeights(model, config['Hyperparameters']['pretrained_weights'], device)
        except Exception as e:
            logger.error('Loading pretrained weights failed - continuing with randomized weights')

    if args.lr_finder:
        find_lr(model_args, dataset)
    elif args.inference:
        # add gradcam
        if args.images and not args.no_gradcam:
            model = add_gradcam(model, multimodal=multimodal)
        
        if args.survival:
            inference_survival(model_args, model, dataset, device)
        else:
            inference(model_args, model, dataset, device)
    else:

        train(model_args, model, dataset, device)

