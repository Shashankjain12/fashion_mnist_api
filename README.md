# FASHION MNIST API
This is the api for prediction of fashion mnist based on the data provided to it.
This model will return you the prediction of image which will predict the class of fashion items by using CNN.

Here is the classes which sepearates the identification of model
{
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

 

API for fashion mnist dataset
Steps for training  of your model based on the dataset
## Step 1 :- Train your model
Our scripts for training is written under py/ folder of the repository
Takes in two arguments ie. path for training the images and testing of the images

```
   python py/train_mnist.py --train fashion/ --test fashion/
```

This will generate your model under saved_model/ directory with name fashion_mnist.h5

## Step 2 :- Testing your model

We can test the model by passing the new image which we want to predict and based on the model it will return the name according to the dictionary associated with it

```
   python py/testing_model.py -m saved_model/fashion_mnist.h5 -i tests/sample_image.jpg
```
It will return us the predicted item given as an image to it

