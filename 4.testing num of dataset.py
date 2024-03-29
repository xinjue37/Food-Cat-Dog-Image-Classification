import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Change the name of the dataset to get the dataset
path = os.path.join(os.getcwd(), "test_dataset")
dataset = tf.data.experimental.load(path)
print(dataset)

i = 0
for _,_ in dataset:
     i+=1
print("Num of data =",i)

d=0
c=0
f=0

# Testing for the image and label
for image,label in dataset.take(4000):
    # print(image)
    if label == [1]:
        d+=1
    elif label == [0]:
        c+=1
    elif label == [2]:
        f+=1

print("c =",c,"d =",d,"f =",f)
