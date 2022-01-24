import os
import imageio
import numpy as np
import pandas as pd
from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, ImageInput
from bentoml.frameworks.sklearn import SklearnModelArtifact
from bentoml.frameworks.pytorch import PytorchModelArtifact
from torchvision import transforms
from models import binary_clf

"""
The @env decorator specifies the dependencies and environment settings required for this prediction service. 
It allows BentoML to reproduce the exact same environment when moving the model and related code to production. 
With the infer_pip_packages=True flag, BentoML will automatically find all the PyPI packages that are used by the prediction service code and pins their versions.

@artifact(...) here defines the required trained models to be packed with this prediction service.
BentoML model artifacts are pre-built wrappers for persisting, loading and running a trained model. 
"""

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model')])

class IrisClassifier(BentoService):
    """
    When the @api batch flag is set to True, an inference APIs is suppose to accept a list of inputs and return a list of results.
    In the case of DataframeInput, each row of the dataframe is mapping to one prediction request received from the client.
    BentoML will convert HTTP JSON requests into pandas.DataFrame object before passing it to the user-defined inference API function.
    """
    @api(input=DataframeInput())
    def predict(self, df: pd.DataFrame):
        """
        An inference API named `predict` with Dataframe input adapter, which codifies
        how HTTP requests or CSV files are converted to a pandas Dataframe object as the
        inference API function input
        """
        return self.artifacts.model.predict(df)

@env(auto_pip_dependencies=True,
     infer_pip_packages=True, pip_packages=['torchvision'])
@artifacts([PytorchModelArtifact('clf_model')])



class CatDogClassifier(BentoService):

    @api(input=ImageInput(accept_image_formats=['.jpg', ".png", ".jpeg"]))
    #@api(input=NumpyNdarray(dtype="float32", enforce_dtype=True), batch=True)
    def predict(self, img):
        covert_to_tensor = transforms.Compose([transforms.ToPILImage(), transforms.Resize((255, 255)), transforms.CenterCrop(224), transforms.ToTensor()])
        img = covert_to_tensor(img)
        prediction = self.artifacts.clf_model(img.unsqueeze(0))
        preds = prediction.data.max(1, keepdim=True)[1]

        if preds.item() == 1:
            result = "Dog"
        else:
            result = 'Cat'

        return {'Final Result': result, "Max_index": preds}
