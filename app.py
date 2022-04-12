from flask import Flask
from flask import request
from flask import render_template
from flask import redirect, url_for
import matplotlib.pyplot as plt
import os
import sys
import glob
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import decode_predictions, preprocess_input

app = Flask(__name__)
classes = ['MODERATE', 'NORMAL-PCR+', 'SEVERE', 'N', 'MILD']

model_path = "model.h5"

app.config["IMAGE_UPLOADS"] = "/static/Upload/"

model = keras.models.load_model(model_path)
model.make_predict_function()
print("Model Loaded. Start Serving...")


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    print(imagefile)
    image_path = imagefile.filename

    class_and_prob = predict(image_path, model)

    # pred_class = decode_predictions(preds, top=1)

    print(class_and_prob)
    classification = '%s (%.2f%%)' % (class_and_prob[0][0], class_and_prob[1][0])



    return render_template("index.html", prediction=classification, uploaded=image_path)

# @app.route('/', methods=['POST'])
# def predict():
#     imagefile = request.files['imagefile']
#     image_path = "../images/" + imagefile.filename
#     imagefile.save(image_path)
#
#     img = image.load_img(image_path, target_size=(224,224))
#     img = image.img_to_array(img)
#     img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
#     img = preprocess_input(img)
#     yhat = model.predict(img)
#     label = decode_predictions(yhat)
#     label = label[0][0]
#
#     classification = '%s (%.2f%%)' % (label[1], label[2]*100)
#
#     return render_template('index.html', prediction=classification)


def predict(filename, model):
    img = image.load_img(filename, target_size=(224, 224))
    img = image.img_to_array(img)
    img = img.reshape(1, 224, 224, 3)

    img = img.astype('float32')
    img = img / 255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(5):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]

    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i] * 100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result, prob_result


if __name__ == '__main__':
    app.run(debug=False)

