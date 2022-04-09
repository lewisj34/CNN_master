import os
from time import sleep
from progress.bar import Bar

def getListofFiles(dirName):
    # create a list of files in directory
    listOfFiles = os.listdir(dirName)
    return listOfFiles

def assert_list(img_list, ann_list, printResults=True):
    print("Checking img_list and ann_list have same file names.")
    with Bar('Processing...') as bar:
        for i in range(len(img_list)):
            assert(img_list[i] == ann_list[i])
            if printResults:
                if (img_list[i] == ann_list[i]):
                    print(" SAME: ",img_list[i], ann_list[i])
                else:
                    print(" ERROR")
                    break
            sleep(0.0002)
            bar.next()
    print("Complete. File names are the same.\n")
def main():
    image_dir = "/home/john/Documents/Datasets/kvasir_merged/images"
    annotation_dir = "/home/john/Documents/Datasets/kvasir_merged/annotations"
    image_list = getListofFiles(image_dir)
    annotatation_list = getListofFiles(annotation_dir)
    
    assert_list(image_list, annotatation_list, True)

if __name__ == "__main__":
    main()