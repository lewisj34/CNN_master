import os 

def renameAnnotaitonsETIS(
    etis_ann_path='/home/john/Documents/Datasets/ETIS/annotations'
):
    """
    just a script to change the curr folder and image name structure that is:
        p1.tif
        p2.tif
        p3.tif
        ...
    to:
        1.tif
        2.tif
        3.tif
    """
    for dir, subdirs, files in os.walk(etis_ann_path):
        for f in files:
            f_new = f[1:]
            os.rename(os.path.join(etis_ann_path, f), os.path.join(etis_ann_path, f_new))


if __name__ == '__main__':
    renameAnnotaitonsETIS()