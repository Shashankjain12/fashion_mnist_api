import argparse
# Initialize parser
parser = argparse.ArgumentParser()

# Adding required  argument
parser.add_argument("-m", "--model",required = True, help = "Pass the model which is saved while training")
parser.add_argument("-i", "--image",required = True, help = "Pass the image to test")
args = parser.parse_args()

import warnings
warnings.filterwarnings("ignore")
#Testing the model out on real data
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
model = load_model(args.model)

def load_image_prediction(filename):
    img_dict = {
    0 : "T-shirt/top",
    1 : "Trouser",
    2 : "Pullover",
    3 : "Dress",
    4 : "Coat",
    5 : "Sandal",
    6 : "Shirt",
    7 : "Sneaker",
    8 : "Bag",
    9 : "Ankle boot"
    } 
    # load the image
    img = load_img(filename, grayscale= True,target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    y_pred = model.predict(img)
    classes_x = np.argmax(y_pred, axis=1)
    class_pred = img_dict[classes_x[0]]
    return class_pred

pred = load_image_prediction(args.image)
print(f"Prediction of image is: {pred} ")
