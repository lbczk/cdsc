from read import Reader
from models import model
from keras.preprocessing.image import ImageDataGenerator
from generator import BSONIterator


r = Reader(full_prep=True)

# need the generator
train_bson_file = open(r.train_bson_path, "rb")

# num_classes = len(train_offsets_df["category_id"].unique())
num_classes = 5270
print "Total number of categories in train {}".format(num_classes)


num_train_images = len(r.train_images_df)
num_val_images = len(r.val_images_df)
batch_size = 16

# Tip: use ImageDataGenerator for data augmentation and preprocessing.
train_datagen = ImageDataGenerator()
train_gen = BSONIterator(train_bson_file, r.train_images_df, r.train_offsets_df,
                         num_classes, train_datagen, batch_size=batch_size, shuffle=True)

val_datagen = ImageDataGenerator()
val_gen = BSONIterator(train_bson_file, r.val_images_df, r.train_offsets_df,
                       num_classes, val_datagen, batch_size=batch_size)

# next(train_gen)  # warm-up

# %time bx, by = next(train_gen)
# plt.imshow(bx[-1].astype(np.uint8))


test_bson_file = open(r.test_bson_path, "rb")

test_datagen = ImageDataGenerator()
test_gen = BSONIterator(test_bson_file, r.test_images_df, r.test_offsets_df,
                        num_classes, test_datagen, batch_size=batch_size,
                        with_labels=False, shuffle=False, keep_indices=True)

# To train the model:
print("fitting\n")
steps_per_epoch = num_train_images // batch_size
model.fit_generator(train_gen, steps_per_epoch=steps_per_epoch, epochs=1, validation_data=val_gen, validation_steps=num_val_images, workers=8)

num_test_samples = len(r.test_images_df)
print("predicting")
predictions = model.predict_generator(test_gen, steps=1 + num_test_samples // batch_size, workers=8)

