from imageai.Detection import ObjectDetection
import os
import pickle
from matplotlib import image
import numpy as np

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
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, path), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
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


print(bottle_type("deneme.jpg", "plastic"))