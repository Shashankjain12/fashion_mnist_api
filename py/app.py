# using flask_restful
from flask import Flask, jsonify, request
from flask.helpers import make_response
from flask_restx import Resource, Api, fields, reqparse
import werkzeug
from flask import Blueprint, request
from flask_restx.namespace import Namespace
import os
import cv2
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

app=Flask(__name__)
blueprint=Blueprint("api",__name__,url_prefix="/api")


api=Api(blueprint,title='Fashion MNIST Machine Learning',
    version='1.0',
    description='Application for Machine learning application model prediction of fashion mnist data'
)
# api=Api(blueprint,doc="/documentation")

app.register_blueprint(blueprint=blueprint)
blueprint = Blueprint('api', __name__)

ns1 =Namespace('fashion_mnist', description='fashion mnist model prediction')
api.add_namespace(ns1)

model_fields = api.model('Model', {
    'prediction': fields.Integer,
    'probability': fields.Float,
    'model': fields.String,
})

# making a class for a particular resource
# the get, post methods correspond to get and post requests
# they are automatically mapped by flask_restful.
# other methods include put, delete, etc.
# another resource to calculate the square of a number

parser = reqparse.RequestParser()
parser.add_argument('file',
                    type=werkzeug.datastructures.FileStorage,
                    location='files',
                    required=True,
                    help='provide a file')


def load_image_prediction(filename, model):
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
    img_prob = y_pred[0][classes_x[0]]*100.0
    return class_pred, img_prob

@ns1.route('/predict/')
@api.expect(parser)
class Model(Resource):
    @api.doc(responses={200: 'OK', 400: 'Invalid Argument', 500: 'Mapping Key Error'},
        description='predict the model')
    def post(self):
        args = parser.parse_args()
        file = args['file']
        if not os.path.isdir("images"):
            os.mkdir("images")
        files = os.listdir("images/")
        file_count = len(files)
        if os.path.splitext(file.filename)[1] in [".jpg", '.jpeg', '.png']:
            filename = os.path.splitext(file.filename)[0]+"_"+str(file_count)+os.path.splitext(file.filename)[1]
            file_write = os.path.join("images", filename)
            file.save(file_write)
            model = load_model("saved_model/fashion_mnist.h5")
            predictions, probability = load_image_prediction(file_write, model)
            return make_response(jsonify({"Output_filename": filename, "prediction" : predictions, "probability": str(probability)+"%"}), 200)
        else:
            return make_response(jsonify(error = "Please enter the correct image file"), 400)

  

# driver function
if __name__ == '__main__':
    app.run(debug = True)
