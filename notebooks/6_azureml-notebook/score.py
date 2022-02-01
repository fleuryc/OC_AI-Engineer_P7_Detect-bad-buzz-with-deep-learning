import os
import json
import numpy as np
import tensorflow as tf


def init():
    global model

    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model/data/model")
    model = tf.keras.models.load_model(model_path)



def run(raw_data):
    data = np.array(json.loads(raw_data)["data"])
    y_hat = model.predict(data)

    return y_hat.tolist()
