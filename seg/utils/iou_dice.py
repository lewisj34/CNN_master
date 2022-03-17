import sys 
import torch.nn.functional as F
import numpy as np

eps = 0.00002

def precision(tp, fp):
    return tp / (tp + fp + eps)

def recall(tp, fn):
    return tp / (tp + fn + eps)

def mean_precision(gt, pred): 
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(pred)): 
        if gt[i]==pred[i]==1:
           tp += 1
        if pred[i]==1 and gt[i]!=pred[i]:
           fp += 1
        if gt[i]==pred[i]==0:
           tn += 1
        if pred[i]==0 and gt[i]!=pred[i]:
           fn += 1
    precision_val = tp / (tp + fp)
    return precision_val

def mean_recall(gt, pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(pred)): 
        if gt[i]==pred[i]==1:
           tp += 1
        if pred[i]==1 and gt[i]!=pred[i]:
           fp += 1
        if gt[i]==pred[i]==0:
           tn += 1
        if pred[i]==0 and gt[i]!=pred[i]:
           fn += 1
    recall_val = tp / (tp + fn)
    return recall_val 

def mean_iou_score(ground_truth, prediction, **kwargs):
    """
    Compute average iou score for output segmetnation map, for a single batch.
    Args:
        @ground_truth: ground truth / mask (NCHW: 1 * 1 * H * W)
        @prediction: predicted segmentation map from model (NCHW: 1 * 1 * H * W)
    """
    axes = (0, 1) 
    intersection = np.sum(np.abs(prediction * ground_truth), axis=axes) 
    mask_sum = np.sum(np.abs(ground_truth), axis=axes) + np.sum(np.abs(prediction), axis=axes)
    union = mask_sum  - intersection 
    
    smooth = .001
    iou = (intersection + smooth) / (union + smooth)
    return iou


def mean_dice_score(ground_truth, prediction, **kwargs):
    """
    Compute average dice score for output segmetnation map, for a single batch.
    Args:
        @ground_truth: ground truth / mask (NCHW: 1 * 1 * H * W)
        @prediction: predicted segmentation map from model (NCHW: 1 * 1 * H * W)
    """
    axes = (0, 1) # W,H axes of each image
    intersection = np.sum(np.abs(prediction * ground_truth), axis=axes) 
    mask_sum = np.sum(np.abs(ground_truth), axis=axes) + np.sum(np.abs(prediction), axis=axes)
    
    smooth = .001
    dice = 2*(intersection + smooth)/(mask_sum + smooth)
    return dice

