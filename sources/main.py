from keras.preprocessing.image import ImageDataGenerator
from models import model
from utils import read_bson, make_category_tables, make_test_set, make_val_set
from generator import BSONIterator

import os
import sys
import math
import io
import numpy as np
import pandas as pd
import multiprocessing as mp
import bson
import struct

# %matplotlib inline
import matplotlib.pyplot as plt

import keras
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from collections import defaultdict
from tqdm import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

print "Keras and tf version :{} - {}".format(keras.__version__, tf.__version__)

# used to determine which dataset we work on. ringo is the sample from the website
# jim is an artificial dataset created for the sole purpose of testing efficiency
# janis is the real dataset
dataset_code = "ringo/"

dict_train_test_names = {"ringo": ("train_example.bson", "train_example.bson")}

csv_dir = "../csv/"

# place where the bson files are located
bson_dir = "../bson/"

train_name, test_name = dict_train_test_names[dataset_code]

# train_bson_path = os.path.join(bson_dir, "train_example.bson")
# num_train_products = 7069896

train_bson_path = os.path.join(bson_dir, train_name)
num_train_products = 82

test_bson_path = os.path.join(bson_dir, test_name)
num_test_products = 82

# test_bson_path = os.path.join(bson_dir, "test.bson")
# num_test_products = 1768172

categories_path = os.path.join(csv_dir, dataset_code, "category_names.csv")
categories_df = pd.read_csv(categories_path, index_col="category_id")

# Maps the category_id to an integer index. This is what we'll use to
# one-hot encode the labels.
categories_df["category_idx"] = pd.Series(range(len(categories_df)), index=categories_df.index)

categories_df.to_csv(os.path.join(csv_dir, dataset_code, "categories.csv"))
categories_df.head()

cat2idx, idx2cat = make_category_tables(categories_df)
cat2idx[1000012755], idx2cat[4]

# read bson was here
%time train_offsets_df = read_bson(train_bson_path, num_records=num_train_products, with_categories=True)
%time test_offsets_df = read_bson(test_bson_path, num_records=num_test_products, with_categories=False)

train_offsets_df.to_csv(os.path.join(csv_dir, dataset_code, "train_offsets.csv"))
test_offsets_df.to_csv(os.path.join(csv_dir, dataset_code, "test_offsets.csv"))

cat2idx, idx2cat = make_category_tables(categories_df)


len(train_offsets_df["category_id"].unique())
train_images_df, val_images_df = make_val_set(train_offsets_df, cat2idx, split_percentage=0.2,
                                              drop_percentage=0.)

train_images_df.to_csv(os.path.join(csv_dir, dataset_code, "train_images.csv"))
val_images_df.to_csv(os.path.join(csv_dir, dataset_code, "val_images.csv"))


test_images_df = make_test_set(test_offsets_df)
test_images_df.to_csv(os.path.join(csv_dir, dataset_code, "test_images.csv"))


print("Number of test images:", len(test_images_df))

#  jump here if dataset already build
categories_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "categories.csv"), index_col=0)
cat2idx, idx2cat = make_category_tables(categories_df)

train_offsets_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "train_offsets.csv"), index_col=0)
train_images_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "train_images.csv"), index_col=0)
val_images_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "val_images.csv"), index_col=0)

test_offsets_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "test_offsets.csv"), index_col=0)
test_images_df = pd.read_csv(os.path.join(csv_dir, dataset_code, "test_images.csv"), index_col=0)

print "Total number of images in train {}".format(train_offsets_df["num_imgs"].sum())


# need the generator
train_bson_file = open(train_bson_path, "rb")

# num_classes = len(train_offsets_df["category_id"].unique())
num_classes = 5270
print "Total number of categories in train {}".format(num_classes)


num_train_images = len(train_images_df)
num_val_images = len(val_images_df)
batch_size = 32

# Tip: use ImageDataGenerator for data augmentation and preprocessing.
train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
                         num_classes, train_datagen, batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, val_images_df, train_offsets_df,
                       num_classes, val_datagen, batch_size=batch_size)

# next(train_gen)  # warm-up

# %time bx, by = next(train_gen)
# plt.imshow(bx[-1].astype(np.uint8))


test_bson_file = open(test_bson_path, "rb")

test_datagen = ImageDataGenerator()
test_gen = BSONIterator(test_bson_file, test_images_df, test_offsets_df,
                        num_classes, test_datagen, batch_size=batch_size,
                        with_labels=False, shuffle=False, keep_indices=True)

# To train the model:
print("fitting\n")
%time model.fit_generator(train_gen, steps_per_epoch=num_train_images // batch_size, epochs=1, validation_data=val_gen, validation_steps=num_val_images, workers=8)

num_test_samples = len(test_images_df)
print("predicting")
%time predictions = model.predict_generator(test_gen, steps=1 + num_test_samples // batch_size, workers=8)

# aa = BSONIterator(test_bson_file, test_images_df, test_offsets_df,
#                         num_classes, test_datagen, batch_size=batch_size,
#                         with_labels=False, shuffle=False)

# aa = BSONIterator(train_bson_file, train_images_df, train_offsets_df,
#                          num_classes, train_datagen, batch_size=batch_size, shuffle=False, keep_indices=True)
# bx, by = next(aa)
# plt.imshow(bx[3].astype(np.uint8))
# aa.indices
# bx = next(aa)


