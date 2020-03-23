from builtins import print
from flask import Response
import jsonpickle
import cv2
from future.types.newrange import range_iterator
from imageai.Detection import ObjectDetection
import os
import pickle
import ast
from matplotlib import image
import numpy as np
from flask import Flask, request

app = Flask(__name__)

app.secret_key = "secret key"

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

filename = 'model_glass_vs_tin.sav'
glass_vs_tin_model = pickle.load(open(filename, 'rb'))
filename = 'model_plastic_vs_glass.sav'
glass_vs_plastic_model = pickle.load(open(filename, 'rb'))
filename = 'model_plastic_vs_tin.sav'
plastic_vs_tin_model = pickle.load(open(filename, 'rb'))

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "../dataset/resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()

def bottle_detection(path):
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, path),
                                                 output_image_path=os.path.join(execution_path, "imagenew.jpg"))
    for eachObject in detections:
        if "bottle" == eachObject["name"]:
            if eachObject["percentage_probability"] > 50:
                return True
    return False


'''
    :parameter path: Directory of bottle image
    :parameter bottle_type: Type of given bottle. Types: plastic/glass/tin
'''


def bottle_type(path, bottle_type):
    if bottle_detection(path):
        image_data = [image.imread(path)]
        image_X = np.array(image_data);
        if bottle_type == 'plastic':
            if (plastic_vs_tin_model.predict(image_X) < 0.5) and (glass_vs_plastic_model.predict(image_X) > 0.5):
                return True
        elif bottle_type == 'glass':
            if (glass_vs_tin_model.predict(image_X) < 0.5) and (glass_vs_plastic_model.predict(image_X) < 0.5):
                return True
        elif bottle_type == 'tin':
            if (glass_vs_tin_model.predict(image_X) > 0.5) and (plastic_vs_tin_model.predict(image_X) > 0.5):
                return True
    return False


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# route http posts to this method
@app.route('/plastic', methods=['POST'])
def plastic():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    im = cv2.imread("image.jpg", cv2.IMREAD_UNCHANGED)
    result = bottle_type("image.jpg", "plastic")
    # build a response dict to send back to client
    response = {'message': result
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# route http posts to this method
@app.route('/glass', methods=['POST'])
def glass():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    result = bottle_type("image.jpg", "glass")
    # build a response dict to send back to client
    response = {'message': result
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# route http posts to this method
@app.route('/tin', methods=['POST'])
def tin():
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("image.jpg", img)
    result = bottle_type("image.jpg", "tin")
    # build a response dict to send back to client
    response = {'message': result
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='192.168.1.16', threaded=False)
