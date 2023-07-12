import tensorflow as tf
import os


def combine_dataset(dataset_name1, dataset_name2, saved_name):
    path_food = os.path.join(os.getcwd(), dataset_name1)
    path_cat_dog = os.path.join(os.getcwd(), dataset_name2)

    food_dataset = tf.data.experimental.load(path_food)
    cat_dog_dataset = tf.data.experimental.load(path_cat_dog)

    dataset = food_dataset.concatenate(cat_dog_dataset)
    dataset.shuffle(1000)  # 這東西shuffle 了個寂寞啊

    path = os.path.join(os.getcwd(), saved_name)
    tf.data.experimental.save(dataset, path)


"""For train 18610(Dog_Cat) + 30300(Food) = (48910 image)"""
combine_dataset("food_train", "dog_cat_train", "train_dataset")

"""For test 2326(Dog_Cat) + 3788(Food) = (6114 image)"""
combine_dataset("food_test", "dog_cat_test", "test_dataset")

"""For Validation 2326(Dog_Cat) + 3787(Food) = (6113 image)"""
combine_dataset("food_validation", "dog_cat_validation", "valid_dataset")
