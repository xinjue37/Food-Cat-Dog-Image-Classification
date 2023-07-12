import tensorflow_datasets as tfds
import os
import tensorflow as tf

(train, test, validation), _ = tfds.load('cats_vs_dogs',
                                         split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
                                         with_info=True, as_supervised=True)


# return as image that is reshaped to Image_size
def format_image(image, label):
    image = tf.cast(image, tf.float32)  # To make
    image = (image / 127.5) - 1  # Make the data range between [-1,1] as the MobileNetV2 need image scale range [-1,1]
    image = tf.image.resize(image, (160, 160))  # Fix all the shape to be (160,160,3) for each image
    return image, label


train = train.map(format_image)
validation = validation.map(format_image)
test = test.map(format_image)

# Convert all the data to <BatchDataset shapes: ((None, 160, 160, 3), (None,)), types: (tf.float32, tf.int64)>
# So it will be accepted by the MobileNetV2
batch_size = 1
train_batches = train.batch(batch_size)
validation_batches = validation.batch(batch_size)
test_batches = test.batch(batch_size)

# Save the dataset
path = os.path.join(os.getcwd(), "dog_cat_train")
tf.data.experimental.save(train_batches, path)

path = os.path.join(os.getcwd(), "dog_cat_validation")
tf.data.experimental.save(validation_batches, path)

path = os.path.join(os.getcwd(), "dog_cat_test")
tf.data.experimental.save(test_batches, path)
