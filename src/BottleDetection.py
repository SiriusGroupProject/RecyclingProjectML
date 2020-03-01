from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(os.path.join(execution_path, "..\\dataset\\resnet50_coco_best_v2.0.1.h5"))
detector.loadModel()


def define_model(path):
    detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path, path), output_image_path=os.path.join(execution_path , "imagenew.jpg"))
    for eachObject in detections:
        if "bottle" == eachObject["name"]:
            if eachObject["percentage_probability"] > 50:
                return True
    return False