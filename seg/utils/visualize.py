import torch 
import random 
import numpy as np
import matplotlib.pyplot as plt 

def plot_test_valid_loss(
    test_loss_list: list, 
    valid_loss_list: list, 
    train_loss_list: list,
    num_epochs: int, 
    save_dir='results/', 
    title='Loss Curve', 
    save_name='loss_curve.png',
    showPlot=False,
):
    '''
    Plots the loss curve for test and valid losses. Losses must be in the form 
    of a list. 
        @test_loss_list: the list containing the test losses across the number
        of epochs
        @valid_loss_list: the list containing the validation losses across tghe
        number of epochs 
        @num_epochs: the number of epochs corresponding to training epochs 
        @save_dir: where to save the results pic 
    '''
    epoch_list = list()
    for i in range(num_epochs): 
        epoch_list.append(i + 1)
    # print(f'len(test_loss_list, valid_loss_list, epoch_list): {len(test_loss_list), len(valid_loss_list), len(epoch_list)}')
    assert len(test_loss_list) == len(valid_loss_list) == len(train_loss_list) == len(epoch_list)
    test_losses = np.array(test_loss_list)
    valid_losses = np.array(valid_loss_list)
    train_losses = np.array(train_loss_list)
    epochs = np.array(epoch_list)

    plt.plot(epochs, test_losses, label = 'test loss')
    plt.plot(epochs, valid_losses, label = 'valid loss')
    plt.plot(epochs, train_losses, label = 'train loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # plt.ylim(min(min(test_losses), min(valid_losses)), max(max(test_losses), max(valid_losses)))
    plt.title(title)
    plt.legend()
    plt.savefig(f'{save_dir}/' + save_name)

    if showPlot:
        plt.show()
    else:
        plt.close()



def visualizeModelOutputfromDataLoader(
    dataloader, 
    model,
    num_batches_to_show=2,    
    save_dir='results/',
    title='Model Output and Visualization',
    save_name='model_visualization.png',
    showPlot=False,
):
    '''
    THIS IS THE FINAL VERSION. 

    Takes in a dataloader (batch_size == 1 MUST), as well as a model (no 
    requirements on whether or not this thing is trained but obviously you
    would probably want it to  be) and shows the number of batches or the number
    of images to show,  (since batch_size = 1)  
        @dataloader: the dataloader containing the image and mask data
        @model: the model to generate the output segementation map 
        @num_batches_to_show: the number of images (or batches) to show 

    '''
    model.eval()

    assert dataloader.batch_size == 1, 'batch_size must be == 1'
    assert num_batches_to_show <= len(dataloader)

    image_list = list()
    gt_list = list()
    pred_list = list() 

    for i, image_gt in enumerate(dataloader):
        images, images_for_visualizaition, gts = image_gt

        images = images.cuda()
        images_for_visualizaition = images_for_visualizaition.cuda()
        gts = gts.cuda() 

        with torch.no_grad():
            output = model(images)
        
        output = output.squeeze(0).squeeze(0).sigmoid().data.cpu().numpy()
        gts = gts.squeeze(0).squeeze(0).data.cpu().numpy()

        # gts = 1 * (gts > 0.5)
        output = 1 * (output > 0.5)

        images = images.permute(0, 2, 3, 1)
        images = images.squeeze(0).data.cpu().numpy()
        images_for_visualizaition = images_for_visualizaition.permute(0, 2, 3, 1)
        images_for_visualizaition = images_for_visualizaition.squeeze(0).data.cpu().numpy()

        # output of images, gts, and output now: 
        #       ((256, 256, 3), (256, 256), (256, 256))

        image_list.append(images_for_visualizaition)
        gt_list.append(gts)
        pred_list.append(output)
    
    assert(len(image_list) == len(gt_list) == len(pred_list))
    sample = random.sample(range(len(image_list)), num_batches_to_show) 
    
    fig, ax = plt.subplots(num_batches_to_show, 3, sharex='col', sharey='row')

    for i in range(len(sample)):
        ax[i, 0].imshow(image_list[sample[i]])
        ax[0, 0].set_title(f'Input Image')
        ax[i, 1].imshow(pred_list[sample[i]], cmap='gist_gray')
        ax[0, 1].set_title(f'Model Output')
        ax[i, 2].imshow(gt_list[sample[i]], cmap='gist_gray')
        ax[0, 2].set_title(f'Ground Truth')
    fig.suptitle(title) 
    plt.savefig(f'{save_dir}/' + save_name)
    
    if showPlot:
        plt.show()
    else:
        plt.close()


# def getRandomSampleFromBatch(
#     data_t, # input data tensor (NCHW)
#     mask_t, # input mask tensor (NCHW)
#     num_samples, # number to sample from batch
#     seed=None, 
# ):
#     '''
#     Takes and input tensor, input_t (N, C, H, W), and outputs a random sample 
#     from that tensor, such that output_t (num_samples, C, H, W)
#         @data_t: the data tensor (N, C, H, W)
#         @mask_t: the mask tensor (N, C, H, W)
#         @num_samples: the number of samples to randomly sample from input_t
#     '''
#     if seed is not None: 
#         torch.manual_seed(seed)
#     else: 
#         print(f'Not seeding the visualized output in: {__file__}')
#     N, _, _, _ = data_t.shape
#     perm_batch = torch.randperm(N) # ex output: tensor([2, 1, 3, 0, 4]) 
#     idx = perm_batch[:num_samples] # ex output: tensor([2, 1, 3]) 
#     data_t = data_t[idx] # output shape should be: num_samples, C, H, W
#     mask_t = mask_t[idx]
#     return data_t, mask_t
    

# def defunct_visualize_image_and_GTs(
#     dataloader, 
#     num_shown_in_batch=3,
#     num_batches_to_show=2,
# ):
#     '''
#     Takes in a dataloader loaded with data and shows the input image and its 
#     corresponding ground truth. 
#         @dataloader: the dataloader object containing the inputs and gts
#         @num_shown_in_batch: the number in each batch to show 
#         @num_batches_to_show: the number of batches (new plots) to generate
#     '''
#     print(f'Num shown in each batch: {num_shown_in_batch}')
#     print(f'Num of batches shown in sequential plots: {num_batches_to_show}')

#     # if you've input more batches to show than there are total batches then 
#     # just set the number of batches to show to be the total number of batches 
#     if num_batches_to_show > len(dataloader):
#         num_batches_to_show = len(dataloader) 
    
#     if num_shown_in_batch > dataloader.batch_size:
#         num_shown_in_batch = dataloader.batch_size
#     # print('NEW NUMBER OF BATCHES TO SHOW: ', num_batches_to_show)

#     data_iter = iter(dataloader)
#     for i in range(num_batches_to_show):
#         images, masks = next(data_iter)
#         sample_images, sample_masks = getRandomSampleFromBatch(
#             images, masks, num_shown_in_batch, seed=None)
#         sample_images = np.transpose(sample_images.numpy(), (0, 2, 3, 1))
#         sample_masks = np.transpose(sample_masks.numpy(), (0, 2, 3, 1))        
#         fig, ax = plt.subplots(num_shown_in_batch, 3, sharex='col', sharey='row')
#         for j in range(num_shown_in_batch):
#             ax[j, 0].imshow(sample_images[j, :, :, :])
#             ax[j, 0].set_title(f'{j}: input')
#             ax[j, 1].imshow(sample_masks[j, :, :, :])
#             ax[j, 1].set_title(f'{j}: pred')
#             ax[j, 2].imshow(sample_masks[j, :, :, :])
#             ax[j, 2].set_title(f'{j}: gt')
#         fig.suptitle(f'Batch Number: {i} Samples')
#     plt.show()
