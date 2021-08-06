import numpy as np
from tensorflow import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from configparser import ConfigParser

config = ConfigParser()
config.read("./config.ini")
MODEL_PATH = config["LOCAL"]["MODEL_PATH"]
IMAGE_SIZE = (300, 300)


class VehicleClassification:
    _instance = None
    model = None
    _mappings = [
        "Two-wheeler",
        "Bus",
        "Car"
    ]

    def process_image(self, image_path):
        """ preprocessing of a vehicle image provided by image path before prediction

            :param image_path (str) - path of image under prediction
            :return img_array (ndarray) : image array data
        """

        img = load_img(image_path, target_size=IMAGE_SIZE)
        img_array = img_to_array(img)
        # Create a batch by increase dimensions
        img_array = expand_dims(img_array, 0)
        print(img_array.shape)
        return img_array

    def predict_class(self, image_path):
        """ Predict class of a vehicle image provided by image path

            :param image_path (str): path of image under prediction
            :return class of vehicle (str): a class from _mappings
        """

        img_array = self.process_image(image_path)
        predictions = self.model.predict(img_array)
        vehicle = self._mappings[np.argmax(predictions)]
        return vehicle


def classification_engine(model_path=MODEL_PATH):
    """ Factory function for VehicleClassification class

        :param model_path (str): path of serialised model
        :return VehicleClassification._instance (ndarray):
    """

    if VehicleClassification._instance is None:
        VehicleClassification._instance = VehicleClassification()
        VehicleClassification.model = load_model(model_path)
    return VehicleClassification._instance


if __name__ == "__main__":
    vcs = classification_engine(MODEL_PATH)
    # make a prediction
    keyword = vcs.predict_class("./temp/b1.jpeg")
    print(keyword)
