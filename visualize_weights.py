# import matplotlib.pyplot as plt 
# import random 
# import numpy as np 
# import torch 

# def visualizeWeights(
#     dataloader, 
#     model,
#     num_batches_to_show=2,    
#     save_dir='results/',
#     title='Model Output and Visualization',
#     save_name='model_visualization.png',
#     showPlot=False,
# ):
#     '''
#     THIS IS THE FINAL VERSION. 

#     Takes in a dataloader (batch_size == 1 MUST), as well as a model (no 
#     requirements on whether or not this thing is trained but obviously you
#     would probably want it to  be) and shows the number of batches or the number
#     of images to show,  (since batch_size = 1)  
#         @dataloader: the dataloader containing the image and mask data
#         @model: the model to generate the output segementation map 
#         @num_batches_to_show: the number of images (or batches) to show 

#     '''
#     model.eval()

#     assert dataloader.batch_size == 1, 'batch_size must be == 1'
#     assert num_batches_to_show <= len(dataloader)

#     image_list = list()
#     gt_list = list()
#     pred_list = list() 

#     for i, image_gt in enumerate(dataloader):
#         images, images_for_visualizaition, gts = image_gt

#         images = images.cuda()
#         images_for_visualizaition = images_for_visualizaition.cuda()
#         gts = gts.cuda() 

#         with torch.no_grad():
#             output = model(images)
        
#         output = output.squeeze(0).squeeze(0).sigmoid().data.cpu().numpy()
#         gts = gts.squeeze(0).squeeze(0).data.cpu().numpy()

#         # gts = 1 * (gts > 0.5)
#         output = 1 * (output > 0.5)

#         images = images.permute(0, 2, 3, 1)
#         images = images.squeeze(0).data.cpu().numpy()
#         images_for_visualizaition = images_for_visualizaition.permute(0, 2, 3, 1)
#         images_for_visualizaition = images_for_visualizaition.squeeze(0).data.cpu().numpy()

#         # output of images, gts, and output now: 
#         #       ((256, 256, 3), (256, 256), (256, 256))

#         image_list.append(images_for_visualizaition)
#         gt_list.append(gts)
#         pred_list.append(output)
    
#     assert(len(image_list) == len(gt_list) == len(pred_list))
#     sample = random.sample(range(len(image_list)), num_batches_to_show) 
    
#     fig, ax = plt.subplots(num_batches_to_show, 3, sharex='col', sharey='row')

#     for i in range(len(sample)):
#         ax[i, 0].imshow(image_list[sample[i]])
#         ax[0, 0].set_title(f'Input Image')
#         ax[i, 1].imshow(pred_list[sample[i]], cmap='gist_gray')
#         ax[0, 1].set_title(f'Model Output')
#         ax[i, 2].imshow(gt_list[sample[i]], cmap='gist_gray')
#         ax[0, 2].set_title(f'Ground Truth')
#     fig.suptitle(title) 
#     plt.savefig(f'{save_dir}/' + save_name)
    
#     if showPlot:
#         plt.show()
#     else:
#         plt.close()



import numpy as np
import matplotlib.pyplot as plt
from torchvision import utils

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
    n,c,w,h = tensor.shape

    if allkernels: tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))


if __name__ == "__main__":
    layer = 1
    filter = model.features[layer].weight.data.clone()
    visTensor(filter, ch=0, allkernels=False)

    plt.axis('off')
    plt.ioff()
    plt.show()