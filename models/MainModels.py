import json
import os

import tensorflow as tf


class BinaryClassificationModel(tf.keras.Model):
    def __init__(self, hidden_size=128):
        super(BinaryClassificationModel, self).__init__()

        self.hidden_size = hidden_size

        self.hidden_layer = tf.keras.layers.Dense(self.hidden_size, activation="relu")
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, x, training=False, **kwargs):
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x

    # Code for saving and loading models.
    # This code is absolutely necessary to use the trainer.
    def get_config(self):
        return {"hidden_size": self.hidden_size}

    def _get_sample_data(self):
        sample_data = {"x": tf.constant([[0, 1]], dtype=tf.int32)}
        return sample_data

    def save(self, save_dir):
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(self.get_config(), f)

        self(**self._get_sample_data())
        self.save_weights(os.path.join(save_dir, "model_weights.h5"))

        return os.listdir(save_dir)

    @classmethod
    def load(cls, save_dir):
        with open(os.path.join(save_dir, "config.json"), "r") as f:
            config = json.load(f)

        model = cls(**config)
        model(**model._get_sample_data())
        model.load_weights(os.path.join(save_dir, "model_weights.h5"))

        return model
