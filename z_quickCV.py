import os 
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt 

from tqdm import tqdm 
from time import sleep 



def get_max_dims(dir):
    file_list = os.listdir(dir)
    max_dim_x = 0
    max_dim_y = 0 
    min_dim_x = 0
    min_dim_y = 0 
    max_dims = (0, 0)
    unique_dims = list()

    min_dims = cv2.imread(dir + f'/{file_list[0]}', cv2.IMREAD_COLOR).shape

    for i in tqdm(range(len(file_list))):
        sleep(0.001)
        full_path = dir + f'/{file_list[i]}'
        img = cv2.imread(full_path, cv2.IMREAD_COLOR)
        # print(f'name: {file_list[i]}')

        if img.shape[0] * img.shape[1] > max_dims[0] * max_dims[1]: 
            max_dims = img.shape
            # print(f'new max_dims: {max_dims}')
        if img.shape[0] * img.shape[1] < min_dims[0] * min_dims[1]:
            min_dims = img.shape
            # print(f'new min_dims: {min_dims}')
        
        if img.shape[0] > max_dim_x:
            max_dim_x = img.shape[0]
        if img.shape[1] > max_dim_y:
            max_dim_y = img.shape[1]
        if img.shape[0] < min_dim_x:
            min_dim_x = img.shape[0]
        if img.shape[1] < min_dim_y:
            min_dim_y = img.shape[1]
        if img.shape not in unique_dims:
            unique_dims.append(img.shape)
        
        if dir == r'C:\Users\johni\OneDrive\Datasets\master_polyp\TrainDataset\image' and file_list[i][0] == 'c': # for combined set, find what the characteristics of the CVC-ClinicDB part of it are 
            break
    
    print(f'max_dims: {max_dims}')

    print(f'min_dims: {min_dims}')
    print(f'unique_dims: {unique_dims}')

def visualizeImageAndGroundTruth(
    master_path, 
    save_dir='./z/imgs',
    title='Model Output and Visualization',
    save_name='model_visualization.png',
    showPlot=True,
):
    '''
    Takes in list of file_paths @file_paths, and generates a plot of all the 

    '''
    os.makedirs(save_dir, exist_ok=True)

    img_path = master_path + '/imgs/'
    ann_path = master_path + '/anns/'
    
    img_paths = os.listdir(img_path)
    ann_paths = os.listdir(ann_path)

    # list that holds the cv2 objects 
    img_list = list()
    ann_list = list()

    assert len(img_paths) == len(ann_paths), \
        f'len(img_paths): {len(img_paths)} != len(ann_paths): {len(ann_paths)}'


    nrow = 5
    ncol = 2

    fig = plt.figure(figsize=(ncol+1, nrow+1)) 

    gs = gridspec.GridSpec(nrow, ncol,
            wspace=0.05, hspace=0.05, 
            top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
            left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

    for i in range(nrow):
        
        for j in range(ncol):
            if j == 0:
                img = cv2.imread(master_path + f'/imgs/{img_paths[i]}', cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
                ax= plt.subplot(gs[i,j])
                ax.imshow(img)
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            if j ==1:
                ann = cv2.imread(master_path + f'/anns/{ann_paths[i]}', cv2.IMREAD_GRAYSCALE)
                ann = cv2.resize(ann, (512, 512), interpolation=cv2.INTER_AREA)
                # im = np.random.rand(512,512)
                ax = plt.subplot(gs[i,j])
                ax.imshow(ann, cmap='gist_gray')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            
            if i == 0:
                ax.set(ylabel='Kvasir')
                ax.label_outer()
            elif i == 1:
                ax.set(ylabel='CVC-ClinicDB')
                ax.label_outer()
            elif i == 2:
                ax.set(ylabel='CVC-ColonDB')
                ax.label_outer()
            elif i == 3:
                ax.set(ylabel='ETIS')
                ax.label_outer()
            elif i == 4:
                ax.set(ylabel='EndoScene')
                ax.label_outer()
            else:
                raise ValueError('askdjfh')

            # ax.axes.xaxis.set_visible(False)
            # ax.axes.yaxis.set_visible(False)
            plt.xticks([])
            plt.yticks([])


    plt.show()

if __name__ == '__main__':
    # print(f'Evaluating Kvasir dataset:')
    # kvasir = r"C:\Users\johni\OneDrive\Datasets\Kvasir-SEG\images"
    # get_max_dims(kvasir)
    
    # print(f'Evaluating CVC-ClinicDB dataset in the Training set of Combined Kvasir and CVC-ClinicDB')
    # comb_kvasir_clincdb_dir = r"C:\Users\johni\OneDrive\Datasets\master_polyp\TrainDataset\image"
    # get_max_dims(comb_kvasir_clincdb_dir)

    # print(f'Evaluating test CVC-ClinicDB dataset in the master_polyp test')
    # cvc_clinicDB_test = r"C:\Users\johni\OneDrive\Datasets\master_polyp\TestDataset\CVC-ClinicDB\images"
    # get_max_dims(cvc_clinicDB_test)

    # print(f'Evaluating test CVC-ColonDB dataset in the master_polyp test')
    # cvc_colonDB_test = r"C:\Users\johni\OneDrive\Datasets\master_polyp\TestDataset\CVC-ColonDB\images"
    # get_max_dims(cvc_colonDB_test)

    # print(f'Evaluating test ETIS dataset in the master_polyp test')
    # ETIS_test = r"C:\Users\johni\OneDrive\Datasets\master_polyp\TestDataset\ETIS-LaribPolypDB\images"
    # get_max_dims(ETIS_test)

    # print(f'Evaluating test CVC_300 / EndoScene dataset in the master_polyp test')
    # CVC_300_test = r"C:\Users\johni\OneDrive\Datasets\master_polyp\TestDataset\CVC-300\images"
    # get_max_dims(CVC_300_test)

    tiny_master_path = r'C:/Users/johni/OneDrive/Datasets/tiny_polyp/'
    visualizeImageAndGroundTruth(
        master_path=tiny_master_path,
    )