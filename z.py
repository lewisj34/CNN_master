"""
GOAL OF THIS FILE IS TWO-FOLD:
    1. Modify input so that we can have transforms variable 
    2. see if we can do something about incoporating the ground truth earlier
        in network 
"""

def main():
    # import data 
    train_loader = get_dataset(
        dataset, 
        save_dir + "/data_train.npy", # location of train images.npy file (str)
        save_dir + "/mask_train.npy" , # location of train masks.npy file (str)
        batchsize=batch_size, 
        normalization="deit" if "deit" in backbone else "vit")



    val_loader = 

    test_loader = 

if __name__ == "__main__":
    main()