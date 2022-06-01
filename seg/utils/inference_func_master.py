import logging
import torch 
import numpy as np



def metric_reporter_general(
    model, 
    loader,
    inferencer,
    loss_fn,
):
    """
    Takes in a data loader and reports all the major statistics generated by 
    our inferencer module. Doesn't matter whether it's `val` or any `test`. 
    """
    model.eval()
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

    logging.info("\tLoss: {:.4f}".format(loss_tensor.mean()))
    logging.info("\tDice: {:.4f}".format(dice_tensor.mean()))
    logging.info("\tIoU: {:.4f}".format(iou_tensor.mean()))
    logging.info("\tAccuracy: {:.4f}".format(acc_tensor.mean()))
    logging.info("\tPrecision: {:.4f}".format(prec_tensor.mean()))
    logging.info("\tRecall: {:.4f}".format(rec_tensor.mean()))
    logging.info("\tTrue positive average: {:.4f}".format(tp_tensor.mean()))
    logging.info("\tFalse positive average: {:.4f}".format(fp_tensor.mean()))
    logging.info("\tTrue negative average: {:.4f}".format(tn_tensor.mean()))
    logging.info("\tFalse negative average: {:.4f}".format(fn_tensor.mean()))

    return loss_tensor.mean(), iou_tensor.mean(), dice_tensor.mean()

def inference_master(
    model, 
    loader, 
    inferencer, 
    loss_fn, 
    test_type,
):
    '''
    Evaluate either test or validation set (for the master dataset).
    Assumes validation set is just a subset of the train set, so has no particular
    focus on any particular dataset (ETIS, Kvasir, CVC's, etc.) 

    Function is called per epoch. 
        @model: the model to run the inference test on 
        @loader: the dataloader containing the test or validation set 
        @inferencer: class that holds all the methods to calculate acc, TP, etc
        @loss_fn: the nn.Module class containing the loss function 
        @test_type: str denoting whether we're working with test or valid set 
    '''

    test_types = ['valid', 'validation', 'test', 'testing']
    assert test_type in test_types, f'{test_type} invalid. Options: {test_types}'

    if test_type == 'valid' or test_type == 'validation':
        print("\nValidation Dataset Statistics: \n")
        logging.info("\nValidation Dataset Statistics: \n")
        meanValidLoss, meanValidIoU, meanValidDice = metric_reporter_general(
            model=model,
            loader=loader,
            inferencer=inferencer,
            loss_fn=loss_fn
        )
        return meanValidLoss, meanValidIoU, meanValidDice 
    else:
        test_cls = ['CVC_300', 'CVC_ClinicDB', 'CVC_ColonDB', 'ETIS', 'Kvasir']

        # create a 5x3 (CVC_300 -> Kvasir)x(Loss, tIoU, Dice) for test scores
        test_results = np.zeros((5, 3))
        
        assert isinstance(loader, list), \
            f'{test_type} but no list of loaders to check against in master set'
        for i in range(len(test_cls)):
            print(f'\n{test_cls[i]} Test Dataset Statistics: \n')
            logging.info(f'\n{test_cls[i]} Test Dataset Statistics: \n')
            test_results[i, :] = metric_reporter_general(
                model=model,
                loader=loader[i],
                inferencer=inferencer,
                loss_fn=loss_fn
            )
        print('\n')
        print(f'Param \t\t{test_cls[4]} \t\t{test_cls[1]} \t{test_cls[2]} \t{test_cls[0]} \t{test_cls[3]} \t\tAvg(dice, IoU)')
        print("avg(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f})".format(
            test_results[4, 2], test_results[4, 1], 
            test_results[1, 2], test_results[1, 1], 
            test_results[2, 2], test_results[2, 1], 
            test_results[0, 2], test_results[0, 1], 
            test_results[3, 2], test_results[3, 1], 
            test_results[:,2].mean(), test_results[:,1].mean()
            )
        )
        logging.info(f'Param \t\t{test_cls[4]} \t\t{test_cls[1]} \t{test_cls[2]} \t{test_cls[0]} \t{test_cls[3]} \t\tAvg(dice, IoU)')
        logging.info("avg(dice, iou) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f}) \t({:.3f}, {:.3f})".format(
            test_results[4, 2], test_results[4, 1], 
            test_results[1, 2], test_results[1, 1], 
            test_results[2, 2], test_results[2, 1], 
            test_results[0, 2], test_results[0, 1], 
            test_results[3, 2], test_results[3, 1], 
            test_results[:,2].mean(), test_results[:,1].mean()
            )
        )
        print('\n')
        return test_results # 5 x 3 tensor

if __name__ == '__main__':
    # import dataset and model, and inferencer module
    from seg.utils.t_dataset import get_tDatasets_master 
    from seg.model.CNN.CNN_backboned import CNN_BRANCH_WITH_BACKBONE
    from seg.utils.inferenceV2 import InferenceModule
    from seg.model.losses.iou_loss import IoULoss

    model = CNN_BRANCH_WITH_BACKBONE(3, 1, 16, 'resnet18', pretrained=True).cuda()
    test_loaders = get_tDatasets_master(
            save_dir='seg/data/master',
            normalize_gt=False,
            batch_size=1, 
            normalization='vit', 
            num_workers=4,
            pin_memory=True,
        )
    inference_master(
        model, 
        loader=test_loaders, 
        inferencer=InferenceModule(eps=0.0001, activation='0-1'), 
        loss_fn=IoULoss(nonlin='sigmoid'), 
        test_type='test',
    )