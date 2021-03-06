import csv 
import os 
from itertools import zip_longest 

from .check_files import getListofFiles, assert_list
def generate_csv(
    img_dir = "/home/john/Documents/Datasets/kvasir_merged/images", 
    ann_dir = "/home/john/Documents/Datasets/kvasir_merged/annotations",
    csv_save_name = "metadata.csv",
    csv_save_dir = "csvs/"
):
    '''
    Generates a csv file of all the images and their associated file paths.
    Returns the file location of that file path. 
    '''
    assert os.path.isdir(img_dir), f'img_dir: {img_dir} DNE'
    assert os.path.isdir(ann_dir), f'ann_dir: {ann_dir} DNE'

    if os.path.isdir(csv_save_dir):
        print(f'save_dir: {csv_save_dir} exists')
        csv_save_name = csv_save_dir + csv_save_name
    else:
        os.mkdir(csv_save_dir)
        csv_save_name = csv_save_dir + csv_save_name

    img_list = getListofFiles(img_dir)
    ann_list = getListofFiles(ann_dir)

    # check that files in each directory, images and annotations are the same
    assert_list(img_list, ann_list, False)

    # modify lists so that they contain the full filepath 
    image_ids = list()
    for i in range(len(img_list)):
        image_ids.append(img_list[i])
        img_list[i] = img_dir + "/" + img_list[i]
        ann_list[i] = ann_dir + "/" + ann_list[i]

    # create csv 
    d = [image_ids, img_list, ann_list]
    export_data = zip_longest(*d, fillvalue = '')
    with open(csv_save_name, 'w', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerow(("image_ids","image_path", "mask_path"))
        wr.writerows(export_data)
    myfile.close()

    return os.path.realpath(myfile.name)

def main():
    generate_csv()

if __name__ == "__main__":
    main()