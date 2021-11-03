# FASHION MNIST API
This is the api for prediction of fashion mnist based on the data provided to it.
This model will return you the prediction of image which will predict the class of fashion items by using CNN.


### Labels
Each training and test example is assigned to one of the following labels:

| Label | Description |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot | 

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

## Step 3 :- Running an Api on the server
Api can be run on server by 

```
   python py/app.py
```

Our api is connected to swagger UI which can be viewed on localhost:5000/api. api is the endpoint where all of the application framework is hosted.

Also to get the predictions of the model hosted one need to pass image on which the identification need to be done. And with the namespace as fashion_mnist/predict where the api endpoint can be executed.

One can execute the curl request by

```
curl -X 'POST' \
  'http://127.0.0.1:5000/api/fashion_mnist/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@sample_image.jpg;type=image/jpeg'
```

And the response body will be where the prediction will be the associated prediction of the image which is received:-
```
{
  "Output_filename": "sample_image_0.jpg",
  "prediction": "Pullover",
  "probability": "99.99914169311523%"
}
```

One can access swagger page to get an interactive way to get the prediction on
```
http://127.0.0.1:5000/api/
```
