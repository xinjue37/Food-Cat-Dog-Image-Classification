import os
import tensorflow as tf
import tensorflow_datasets as tfds
tfds.disable_progress_bar()
import time

"""For the test dataset, it will get approximate 0.96 accuracy 
   in about 144 seconds (difference time based on computer computation ability) """

path = os.path.join(os.getcwd(), "test_dataset")
test_dataset = tf.data.experimental.load(path)

model = tf.keras.models.load_model('Image_classification3.0.h5')

start_time = time.time()

loss0,accuracy0 = model.evaluate(test_dataset)
print(loss0,"accuracy",accuracy0)

end_time = time.time()

print("The time difference =",int(end_time-start_time),"(in seconds)")