import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import shutil
tfds.disable_progress_bar()
from tensorflow.keras.preprocessing import image

# directory 是   /       Not  \

def main():
    create_required_directory()
    dataset = create_dataset()

    if dataset == None:
        print("There is no image in the folder. Please put some images into the folder name 'Image'.")
    else:
        model = tf.keras.models.load_model('Image_classification.h5')  # Load the model
        class_name = ['cat','dog','food']
        prediction = model.predict(dataset)  # Predict the result from the dataset

        # print(prediction)  # The highest value in the list = predicted class
        prediction = prediction.argmax(axis=1)

        num_of_image = 0
        for subdirectory in os.listdir("Image"):
            # Only image file with format ('.jpg', '.png', 'jpeg') can be taken
            if subdirectory.endswith(('.jpg', '.png', 'jpeg')):
                original_path = os.path.join(os.getcwd(), f"Image\{subdirectory}")

                if prediction[num_of_image] == 0:
                    target_path = os.path.join(os.getcwd(), f"Image(After_classify)/Cat_image/{subdirectory}")
                elif prediction[num_of_image] == 1:
                    target_path = os.path.join(os.getcwd(), f"Image(After_classify)/Dog_image/{subdirectory}")
                else:
                    target_path = os.path.join(os.getcwd(), f"Image(After_classify)/Food_image/{subdirectory}")

                num_of_image += 1

                # Move the image to the corresponding directory
                shutil.move(original_path, target_path)

        print("The Classification is done.")




# Return the dataset with <BatchDataset shapes: ((None, 160, 160, 3), (None,)), types: (tf.float32, tf.int64)>
# where (160,160,3) is the image array with range [-1,1] with data type tf.float32
# and (None,) is the label and it is useless in this case as we are not doing verification
def process_image(image_directory):
    # 0 = cat, 1 = dog, 2 = food
    # Simply put the label as need to follow the format for the dataset as the input
    label = tf.constant(1, dtype=tf.int64)

    img = image.load_img(image_directory, target_size=(160, 160))
    image_array = image.img_to_array(img)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = tf.cast(image_array, tf.float32)  # Cast to float type so later can change to [-1,1]
    image_array = (image_array / 127.5) - 1  # Change to [-1,1]
    dataset = tf.data.Dataset.from_tensor_slices((image_array, [label]))  # Change to dataset
    batch_dataset = dataset.batch(1)  # Change to batch dataset so can use as a input for the model
    return batch_dataset


# Return the dataset form by transforming image in the folder "Image"
# If no image found in the folder, return None
def create_dataset():
    num_of_image = 0
    for subdirectory in os.listdir("Image"):
        # Only image file with format ('.jpg', '.png', 'jpeg') can be taken
        if subdirectory.endswith(('.jpg', '.png', 'jpeg')):
            directory = os.path.join(os.getcwd(), f"Image/{subdirectory}")

            if num_of_image == 0:
                dataset = process_image(directory)
            else:
                dataset = dataset.concatenate(process_image(directory))

            num_of_image += 1

    print(f"There are total {num_of_image} images in the folder.")

    if num_of_image != 0:
        return dataset
    else:
        return None


def create_required_directory():
    # 在join 的時候用 '/'
    folder_Image = os.path.join(os.getcwd(), "Image")
    folder_image = os.path.join(os.getcwd(), "Image(After_classify)")
    folder_image_cat = os.path.join(os.getcwd(), "Image(After_classify)/Cat_image")
    folder_image_dog = os.path.join(os.getcwd(), "Image(After_classify)/Dog_image")
    folder_image_food = os.path.join(os.getcwd(), "Image(After_classify)/Food_image")

    # If this path does not exist, create a path for it
    if not os.path.exists(folder_Image):
        os.mkdir(folder_Image)

    # Same as other path, if not exist, create a new one
    if not os.path.exists(folder_image):
        os.mkdir(folder_image)
    if not os.path.exists(folder_image_cat):
        os.mkdir(folder_image_cat)
    if not os.path.exists(folder_image_dog):
        os.mkdir(folder_image_dog)
    if not os.path.exists(folder_image_food):
        os.mkdir(folder_image_food)

main()
