import tensorflow as tf
import tensorflow_datasets as tfds
import time
import os
tfds.disable_progress_bar()


path = os.path.join(os.getcwd(), "train_dataset")
train_dataset = tf.data.experimental.load(path)

path = os.path.join(os.getcwd(), "valid_dataset")
validation_dataset = tf.data.experimental.load(path)

path = os.path.join(os.getcwd(), "test_dataset")
test_dataset = tf.data.experimental.load(path)

base_layers = tf.keras.applications.MobileNetV2(input_shape=(160, 160, 3),
                                                include_top=False,  # don't want load the top layer
                                                weights='imagenet')  # specific weight checkpoints

# Disable the training of the base_model
base_layers.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

# From the internet, we find that add more dense layer can prevent overfitting
# and add some dropout layer that can  randomly sets input units to 0 with
# a frequency of rate at each step during training time, which helps prevent overfitting
dense_layer1 = tf.keras.layers.Dense(320, activation='relu')
dropout_layer1 = tf.keras.layers.Dropout(rate=0.35)
dense_layer2 = tf.keras.layers.Dense(320, activation='relu')
dropout_layer2 = tf.keras.layers.Dropout(rate=0.35)
dense_layer3 = tf.keras.layers.Dense(320, activation='relu')
dropout_layer3 = tf.keras.layers.Dropout(rate=0.35)

# Adding the classifier Dense(3) as we need predict 3 output [ cat(0),dog(1),food(2) ]
dense_layer4 = tf.keras.layers.Dense(3, "softmax")

# Combine the layers together
model = tf.keras.Sequential([base_layers,
                             global_average_layer,
                             dense_layer1, dropout_layer1, dense_layer2, dropout_layer2,
                             dense_layer3, dropout_layer3, dense_layer4])

# Get the summary of the model if you want
model.summary()

learning_rate = 0.00005

# Use Adam because it is the most common use for multiple class image classification
# Use SparseCategoricalCrossentropy because it is suitable for multiple class image classification
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

epoch = 14

# Start training
start_time = time.time()
Training = model.fit(train_dataset,
                     epochs=epoch,
                     validation_data=validation_dataset)

accuracy = Training.history['accuracy']
print("accuracy for training 1:", accuracy)
end_time = time.time()

# Look the training time
total_time = int(end_time - start_time)
print("The time difference for the first train:")
hour = total_time // 3600
minutes = (total_time - hour * 3600) // 60
second = total_time % 60
print(hour, "hour", minutes, "minutes", second, "second")

# Save the model so it can reload anytime without training again
model.save("Image_classification.h5")
