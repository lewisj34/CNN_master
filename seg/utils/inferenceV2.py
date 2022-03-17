'''
This is the final version. 
Contains test() and validate() but runs on the GPU not the CPU 
'''

import torch 
import torch.nn as nn 
from tqdm import tqdm 
import numpy as np 

from seg.utils.t_dataset import get_tDataset
from seg.utils.weighted_loss import weighted_loss
import seg.utils.lovasz_losses as L
from seg.utils.iou_dice import mean_dice_score, mean_iou_score, precision, recall, mean_precision, mean_recall
from seg.utils.accuracy import accuracy

from seg.model.CNN.CNN_backboned import CNN_BRANCH_WITH_BACKBONE

class InferenceModule():
    r"""
    Calculates major model parameter metrics which we report on. 
        @iou: intersection over union 
        @dice: dice score
        @acc: pixel-wise accuracy 
        @prec: precision
        @rec: recall 
        @tp: true positives
        @fp: false positives
        @tn: true negatives 
        @fn: false negatives 

    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion 
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative 
    rate, as specificity/TPR is meaningless in multiclass cases.

    Modified from original version: https://github.com/hsiangyuzhao/Segmentation-Metrics-PyTorch/blob/master/metric.py
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(
        self, 
        gt, 
        pred
    ):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        # get basics 
        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        # calculate major metrics
        iou = tp / (tp + fp + fn)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        prec = (tp + self.eps) / (tp + fp + self.eps)
        rec = (tp + self.eps) / (tp + fn + self.eps)
        
        return iou, dice, acc, prec, rec, tp, fp, tn, fn

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'
        iou, dice, acc, prec, rec, tp, fp, tn, fn = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                          activated_pred)
        return iou, dice, acc, prec, rec, tp, fp, tn, fn
    


def inference(model, loader, inferencer, loss_fn, test_type):
    '''
    Evaluate either test or validation set. Function is called per epoch. 
        @model: the model to run the inference test on 
        @loader: the dataloader containing the test or validation set 
        @loss_fn: the nn.Module class containing the loss function 
        @test_type: str denoting whether we're working with test or valid set 
    '''
    model.eval()

    test_types = ['valid', 'validation', 'test', 'testing']
    assert test_type in test_types, f'Options: {test_types}'


    loss_tensor = torch.zeros(len(loader))
    dice_tensor = torch.zeros(len(loader))
    iou_tensor = torch.zeros(len(loader))
    acc_tensor = torch.zeros(len(loader))
    prec_tensor = torch.zeros(len(loader))
    rec_tensor = torch.zeros(len(loader))
    tp_tensor = torch.zeros(len(loader))
    fp_tensor = torch.zeros(len(loader))
    tn_tensor = torch.zeros(len(loader))
    fn_tensor = torch.zeros(len(loader))

    for i, image_gts in enumerate(loader):
        images, gts = image_gts
        images = images.cuda()
        gts = gts.cuda()


        # need to make sure GTs are [0 or 1], because we both have an option to
        # normalize the ground truth AND ** this is not optional ** but we do
        # gt[i] = gt[i] / 255 when we call a batch every time in __getitem__ in
        # tDataset class. So we MUST execute this following line here. 
        gts = 1 * (gts > 0.5).float() # can't comment this out because in our validation and test loader we have an option to normalize. 

        with torch.no_grad():
            output = model(images)

        # new version
        loss_val = loss_fn(output, gts, smooth = 0.001)

        # call inferencer function via calling the object itself and __call__
        iou, dice, acc, prec, rec, tp, fp, tn, fn = inferencer(gts, output)
        loss_tensor[i] = loss_val
        dice_tensor[i] = dice
        iou_tensor[i] = iou
        acc_tensor[i] = acc
        prec_tensor[i] = prec
        rec_tensor[i] = rec
        tp_tensor[i] = tp
        fp_tensor[i] = fp
        tn_tensor[i] = tn
        fn_tensor[i] = fn
    
    if test_type == 'valid' or test_type == 'validation':
        print("\nValidation Dataset Statistics: \n")
    else:
        print("\nTest Dataset Statistics: \n")
    print("\tLoss: {:.4f}".format(loss_tensor.mean()))
    print("\tDice: {:.4f}".format(dice_tensor.mean()))
    print("\tIoU: {:.4f}".format(iou_tensor.mean()))
    print("\tAccuracy: {:.4f}".format(acc_tensor.mean()))
    print("\tPrecision: {:.4f}".format(prec_tensor.mean()))
    print("\tRecall: {:.4f}".format(rec_tensor.mean()))
    print("\tTrue positive average: {:.4f}".format(tp_tensor.mean()))
    print("\tFalse positive average: {:.4f}".format(fp_tensor.mean()))
    print("\tTrue negative average: {:.4f}".format(tn_tensor.mean()))
    print("\tFalse negative average: {:.4f}".format(fn_tensor.mean()))

    return loss_tensor.mean(), iou_tensor.mean(), dice_tensor.mean()

if __name__ == "__main__":
    model = CNN_BRANCH_WITH_BACKBONE(
        3, 
        1,
        16, 
        'resnet18', 
        pretrained=True,
        with_attention=False,
        with_superficial=False,
        input_size = 256,
    ).cuda()

    data_dir = 'seg/data/kvasir'
    valid_image_root = data_dir + "/data_valid.npy"
    valid_gt_root = data_dir + "/mask_valid.npy"
    test_image_root = data_dir + "/data_test.npy"
    test_gt_root = data_dir + "/mask_test.npy"

    validLoader = get_tDataset(
        image_root = valid_image_root,
        gt_root = valid_gt_root,
        normalize_gt = False,
        batch_size = 1,
        normalization = 'vit',
        num_workers = 4, 
        pin_memory=True,
    )
    testLoader = get_tDataset(
        image_root = test_image_root,
        gt_root = test_gt_root,
        normalize_gt = False,
        batch_size = 1,
        normalization = 'vit',
        num_workers = 4, 
        pin_memory=True,
    )

    from seg.model.losses.iou_loss import IoULoss
    loss_fn = IoULoss(nonlin=None)
    inference(model = model, loader = validLoader, loss_fn=loss_fn, test_type='valid')

    # loss_val = 198273981273

    # model_output = torch.rand(2, 1, 5, 5)
    # ground_truth = torch.rand(2, 1, 5, 5); ground_truth = 1 * (ground_truth > 0.5).float().to(torch.device('cuda'))

    # print(f'model_output: {model_output}')
    # print(f'ground_truth: {ground_truth}')

    # inferencer = InferenceModule(activation='0-1')
    # [iou, dice, acc, prec, rec, tp, fp, tn, fn] = inferencer(ground_truth, model_output)
    # print("\nValidation Dataset Statistics: \n")
    # print("\tLoss: {:.4f}".format(loss_val))
    # print("\tDice: {:.4f}".format(dice))
    # print("\tIoU: {:.4f}".format(iou))
    # print("\tAccuracy: {:.4f}".format(acc))
    # print("\tPrecision: {:.4f}".format(prec))
    # print("\tRecall: {:.4f}".format(rec))
    # print("\tTrue positive average: {:.4f}".format(tp))
    # print("\tFalse positive average: {:.4f}".format(fp))
    # print("\tTrue negative average: {:.4f}".format(tn))
    # print("\tFalse negative average: {:.4f}".format(fn))






# class InferenceModule():
#     def __init__(self, smooth, activation):
#         self.activation = activation 
#         self.smooth = smooth

#     def torch_dice_score(self, inputs, targets, smooth=1):
#             # flatten 
#             inputs = inputs.view(-1)
#             targets = targets.view(-1)
            
#             intersection = (inputs * targets).sum()                            
#             dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
            
#             return dice

#     def torch_dice_score(self, inputs, targets, smooth=1):
#             # flatten 
#             inputs = inputs.view(-1)
#             targets = targets.view(-1)
            
#             # intersection is equivalent to True Positive count
#             # union is the mutually inclusive area of all labels & predictions 
#             intersection = (inputs * targets).sum()
#             total = (inputs + targets).sum()
#             union = total - intersection 
            
#             IoU = (intersection + smooth)/(union + smooth)
                    
#             return IoU

#     def torch_accuracy(self, inputs, targets, smooth=1):
#         """
#         Calculates accuracy (and the relevant params that calculate it: 
#         TP, FP, TN, FN). Assumes sigmoid has been called already and we've got 
#         just an array of probabilities. 
#         """
#         # flatten
#         inputs = inputs.view(-1)
#         targets = targets.view(-1)

#         tp = torch.sum(inputs * targets)  # TP
#         fp = torch.sum(inputs * (1 - targets))  # FP
#         tn = torch.sum((1 - inputs) * (1 - targets))  # TN
#         fn = torch.sum((1 - inputs) * targets)  # FN

#         pixel_acc = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)
#         precision = (tp + smooth) / (tp + fp + smooth)
#         recall = (tp + smooth) / (tp + fn + smooth)

#         return pixel_acc, tp, fp, tn, fn