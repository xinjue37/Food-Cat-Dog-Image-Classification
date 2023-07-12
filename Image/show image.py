import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

image_path = 'dog3.jpg'
img = image.load_img(image_path)
plt.imshow(img)
plt.show()
