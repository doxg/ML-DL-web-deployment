from bento_service import IrisClassifier
from bento_service import CatDogClassifier
from models import IrisModel, binary_clf

class IrisClassifierPacker():
    def __init__(self):
        iris_classifier_service = IrisClassifier()
        iris_classifier_service.pack('model', IrisModel.clf)
        saved_path = iris_classifier_service.save() # Reusable BentoML bundle, which is similar to docker container image.

class CatDogClassifierPacker():
    def __init__(self):
        CatDog_classifier_service = CatDogClassifier()
        CatDog_classifier_service.pack('clf_model', binary_clf.model)
        saved_path = CatDog_classifier_service.save()


packer = CatDogClassifierPacker()
