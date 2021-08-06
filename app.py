from flask import Flask, request, render_template
from vehicle_classification_service import classification_engine
from configparser import ConfigParser
import os
import random

config = ConfigParser()
config.read("./config.ini")
app = Flask(__name__)


@app.route("/")
def nlp_route():
    """ get endpoint for home route

        :return  (jinja template): jinja template to take user input
    """

    return render_template("index.html", size=0)


@app.route("/", methods=["POST"])
def predict_vehicle_class():
    """ post endpoint for home route

        :return prediction  (jinja template[html response]):  predicted vehicle embedded in jinja template
    """

    image_file = request.files["vehicle"]
    image_path = os.path.join("temp", f"{random.randint(1,10000)}.jpg")
    image_file.save(image_path)
    engine = classification_engine()
    prediction = engine.predict_class(image_path)
    os.remove(image_path)
    print(f"identified vehicle is: {prediction}")
    return render_template("index.html", size=len(prediction), result=prediction)


if __name__ == "__main__":
    app.run(port=5000, debug=True)
