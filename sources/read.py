from utils import read_bson, make_category_tables, make_test_set, make_val_set

import os
import pandas as pd
import keras
import tensorflow as tf

from tqdm import *
# Any results you write to the current directory are saved as output.

print("Keras and tf version :{} - {}".format(keras.__version__, tf.__version__))

# used to determine which dataset we work on.
# ringo is the sample from the website
# jim is an artificial dataset created for the sole purpose of testing efficiency
# cass just considers the first 10000 lines of train and test
# janis is the real dataset
dataset_code = "ringo"

dict_train_test_names = {"ringo": ("train_example.bson", "train_example.bson", 82, 82),
                         "cass": ("train.bson", "test.bson", 1e4, 1e4),
                         "janis": ("train.bson", "test.bson", 7069896, 1768172)}
csv_dir = "../csv/"
bson_dir = "../bson/"
train_name, test_name = dict_train_test_names[dataset_code][:2]


class Reader():
    def __init__(self, csv_dir=csv_dir, bson_dir=bson_dir, dataset_code=dataset_code, full_prep=False):
        self.csv_dir = csv_dir
        self.bson_dir = bson_dir
        self.dataset_code = dataset_code
        self.train_name, self.test_name = dict_train_test_names[dataset_code][:2]
        self.train_bson_path = os.path.join(self.bson_dir, self.train_name)
        self.test_bson_path = os.path.join(self.bson_dir, self.test_name)
        self.num_train_products = dict_train_test_names[dataset_code][2]
        self.num_test_products = dict_train_test_names[dataset_code][3]
        self.categories_path = os.path.join(csv_dir, "category_names.csv")
        self.categories_df = pd.read_csv(self.categories_path, index_col="category_id")
        self.categories_df["category_idx"] = pd.Series(range(len(self.categories_df)), index=self.categories_df.index)
        self.cat2idx, self.idx2cat = make_category_tables(self.categories_df)
        self.never_read = True

        if full_prep:
            self.full_prep()

    def read_bson(self):
        if self.never_read:
            self.never_read = False
            self.train_offsets_df = read_bson(self.train_bson_path, num_records=self.num_train_products, with_categories=True)
            self.test_offsets_df = read_bson(self.test_bson_path, num_records=self.num_test_products, with_categories=False)
            self.train_offsets_df.to_csv(os.path.join(self.csv_dir, self.dataset_code, "train_offsets.csv"))
            self.test_offsets_df.to_csv(os.path.join(self.csv_dir, self.dataset_code, "test_offsets.csv"))
            print("Total number of images in train {}".format(self.train_offsets_df["num_imgs"].sum()))

    def make_val_set(self, split_percentage=0.2, drop_percentage=0.):
        if self.never_read:
            self.read_bson()

        train_path = os.path.join(self.csv_dir, self.dataset_code)

        if "train_images.csv" not in os.listdir(train_path):
            print("Calling the make_val_set function")
            self.train_images_df, self.val_images_df = make_val_set(self.train_offsets_df,
                                                                    self.cat2idx, split_percentage=split_percentage,
                                                                    drop_percentage=drop_percentage)
            self.train_images_df.to_csv(os.path.join(self.csv_dir, self.dataset_code, "train_images.csv"))
            self.val_images_df.to_csv(os.path.join(self.csv_dir, self.dataset_code, "val_images.csv"))

        else:
            print("Found the train_images_df.csv file")
            self.train_images_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "train_images.csv"), index_col=0)
            self.val_images_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "val_images.csv"), index_col=0)

    def make_test_set(self):
        print("Doing make test set")
        test_path = os.path.join(self.csv_dir, self.dataset_code)

        if not ("test_images.csv" in os.listdir(test_path)):
            print("Calling the make_test_set function")
            self.test_images_df = make_test_set(test_offsets_df)
            self.test_images_df.to_csv(os.path.join(test_path, "test_images.csv"))
            print("Number of test images:", len(self.test_images_df))

        else:
            print("Found the test_images_df.csv file")
            self.test_images_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "test_images.csv"), index_col=0)

    def full_prep(self):
        self.read_bson()
        self.make_val_set()
        self.make_test_set()
r = Reader(full_prep=True)
# r.read_bson()
# r.make_val_set()
# r.make_test_set()
# r.full_prep()
