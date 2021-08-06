from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.preprocessing import image_dataset_from_directory
from configparser import ConfigParser

config = ConfigParser()
config.read("./config.ini")

DATA_PATH = config["LOCAL"]["DATA_PATH"]
MODEL_PATH = config["LOCAL"]["MODEL_PATH"]
VALIDATION_SPLIT = 0.15
IMAGE_SIZE = (300, 300)
OUTPUT_CLASSES = 3
BATCH_SIZE = 40
SEED = 134
EPOCHS = 10


def preprocess_data(data_path, validation_split=VALIDATION_SPLIT, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE):
    """ preprocessing of dataset

    :param data_path (str)- path of dataset containing various class of images in respective directory
    :param validation_split (float)- fraction of data to be used as validation
    :param image_size (tuple)- height of image,width of image
    :return train dataset, test dataset (tuple):
    """

    train_data_generator = image_dataset_from_directory(
        data_path, validation_split=validation_split, subset="training", seed=SEED, image_size=image_size,
        batch_size=batch_size)

    val_data_generator = image_dataset_from_directory(
        data_path, validation_split=validation_split, subset="validation", seed=SEED, image_size=image_size,
        batch_size=batch_size)
    print("dataset generator created")
    return train_data_generator, val_data_generator


def generate_model(output_classes=OUTPUT_CLASSES, image_size=IMAGE_SIZE):
    """ generate  tensorflow model

    :param output_classes (int): no of classes available
    :param image_size (tuple): height of image,width of image
    :return model (tensorflow.keras.model):
    """

    model: Sequential = Sequential([
        layers.experimental.preprocessing.Rescaling(
            1. / 255, input_shape=(*image_size, 3)),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.1),
        layers.Dense(128, activation="relu"),
        layers.Dense(output_classes)
    ])
    model.compile(optimizer="adam",
                  loss=SparseCategoricalCrossentropy(from_logits=True),
                  metrics=["accuracy"])

    print(model.summary())
    return model


def train_model(model, train_data, val_data, output_path=MODEL_PATH):
    """ train  tensorflow model

    :param data_path (str): path of dataset containing various class of images in respective directory
    :param validation_split (float): fraction of data to be used as validation
    :param output_path (str): path of model for serialization
    :return model (tensorflow.keras.model):
    """

    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS, callbacks=EarlyStopping(patience=2)
    )
    model.save(output_path)
    print(history)
    print("model saved successfully")
    return


if __name__ == "__main__":
    train_data, val_data = preprocess_data(
        DATA_PATH)  # unpack train and test data
    model = generate_model()
    train_model(model, train_data, val_data)
