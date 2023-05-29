from torch.utils.data import DataLoader

import os
import argparse
import shutil
import random
import requests
from pathlib import Path

import tarfile

from utils import get_classes

def split():
    
    # --------------------------------------------------------- DATASET DOWNLOAD AND EXTRACT----------------------------------------------------
    # download the dataset into the data set directory
    with open(ARCHIVE_FILE_PATH, "wb") as f:
            request = requests.get(args.url)
            print("Downloading...")
            f.write(request.content)

    # unzip the dataset into dataset/all directory
    with tarfile.open(ARCHIVE_FILE_PATH) as file:
        print("Unzipping...") 
        file.extractall(ALL_DATASET_PATH)



    # ------------------------------------------------------------ TRAIN TEST SPLIT --------------------------------------------------------------
    # count number of datasets
    number_of_datasets = 0
    for _, _, files in os.walk(IMAGE_PATH):
        number_of_datasets = len(files)

    # create a set of random integers from 1 to number of datasets
    random_set = set()
    random.seed(41)
    start, end = 1, int(args.split * number_of_datasets) 
    for _ in range(start, (end + 4)):
        num = random.randint(1,(number_of_datasets - 1))
        random_set.add(num)
        if len(random_set) == int(end): break

    # create a list conatining anootation files and images
    random_set_iter = iter(random_set)
    annotated_files = []
    image_files = []
    for _, _, files in os.walk(ANNOTATION_PATH):
        annotated_files.extend(files)
    for _, _, files in os.walk(IMAGE_PATH):
        image_files.extend(files)

    # sort images and annotated files
    annotated_files.sort()
    image_files.sort()

    # move image and annotation to test folders
    for rand in random_set_iter:
        shutil.move(os.path.join(ANNOTATION_PATH, annotated_files[rand]),os.path.join(TEST_PATH, annotated_files[rand]))
        shutil.move(os.path.join(IMAGE_PATH, image_files[rand]),os.path.join(TEST_PATH, image_files[rand]))

    # move image and annotation to train folders
    for path in Path(ANNOTATION_PATH).iterdir():
        if path.is_file() and path.suffix == '.xml':
            file_name = str(path).split("/")
            shutil.move(path,os.path.join(TRAIN_PATH, file_name[3]))
    for path in Path(IMAGE_PATH).iterdir():
        if path.is_file():
            file_name = str(path).split("/")
            shutil.move(path,os.path.join(TRAIN_PATH, file_name[3]))


if __name__ == '__main__':
    # arg parser initailizing
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, required= True, help="Download URL of the dataset")
    parser.add_argument("--path", type=str, required= True, help="Dataset directory path")
    parser.add_argument("--split", type=float, required= False, help="Train Test split", default=0.2)
    args = parser.parse_args()

    ARCHIVE_FILE_PATH = os.path.join(args.path, "archive.tar.xz")
    ALL_DATASET_PATH = os.path.join(args.path, "all")
    ANNOTATION_PATH = os.path.join(args.path, "all", "Annotations")
    IMAGE_PATH = os.path.join(args.path, "all", "images")
    TEST_PATH = os.path.join(args.path, "test")
    TRAIN_PATH = os.path.join(args.path, "train")

    RESIZE_TO = 512
    CLASSES = get_classes()

    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    dname_list = dname.split('/')
    root = "/".join(dname_list[:(len(dname_list) - 1)])
    os.chdir(root)

    # split dataset
    split()
