import json
import os

import tensorflow as tf
import transformers
from transformers import PretrainedConfig, TFAutoModel

from models.TransformerLayers import DecoderLayer, EncoderLayer
from models.UtilLayers import PositionalEmbedding


class Transformer(tf.keras.Model):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_size: int,
        code_m: int,
        pe: int = 1000,
        rate: float = 0.1,
    ):
        super(Transformer, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.vocab_size = vocab_size
        self.code_m = code_m
        self.pe = pe
        self.rate = rate

        self.embedding = PositionalEmbedding(vocab_size, embedding_size, pe)
        if embedding_size != hidden_size:
            self.embedding_intermediate = tf.keras.layers.Dense(hidden_size)
        self.encoders = [
            EncoderLayer(hidden_size, num_heads, rate)
            for _ in range(num_encoder_layers)
        ]
        self.code_embedding = tf.keras.layers.Embedding(self.code_m, self.hidden_size)
        self.decoders = [
            DecoderLayer(hidden_size, num_heads, rate)
            for _ in range(num_decoder_layers)
        ]

        self.output_layer = tf.keras.layers.Dense(vocab_size, activation="linear")

    def loss(self, y, pred):
        pred = tf.nn.softmax(pred)
        idx_except_last = tf.meshgrid(
            *[tf.range(s) for s in pred.shape[:-1]], indexing="ij"
        )
        idx = tf.stack(idx_except_last + [y], axis=-1)
        pred_ = tf.gather_nd(pred, idx)

        loss = tf.math.log(pred_)
        mask = tf.cast(tf.math.not_equal(y, 0), dtype=loss.dtype)

        loss = -tf.reduce_sum(loss * mask, axis=-1)
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def dot_attention(self, q, k, v, mask=None):
        logits = tf.matmul(q, k, transpose_b=True)

        if mask is not None:
            logits += tf.cast((1 - mask[:, tf.newaxis, :]), tf.float32) * -1e9

        attention_weights = tf.nn.softmax(logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output

    def create_look_ahead_mask(self, padding_mask):
        size = tf.shape(padding_mask)[-1]

        look_ahead_mask = tf.linalg.band_part(tf.ones((size, size)), -1, 0)

        combine_mask = tf.minimum(
            padding_mask[:, tf.newaxis, tf.newaxis, :],
            tf.cast(look_ahead_mask, padding_mask.dtype),
        )
        return combine_mask

    def call(
        self,
        input_ids=None,
        encoder_embed=None,
        decoder_input_ids=None,
        attention_mask=None,
        decoder_attention_mask=None,
        training=False,
        labels=None,
        **kwargs,
    ):
        # check error
        assert (
            input_ids is not None or encoder_embed is not None
        ), "Either input_ids or encoder_embed must be required."
        assert (
            input_ids is None or encoder_embed is None
        ), "Only one of input_ids and encoder_embed must be entered."

        # encoder
        if input_ids is not None:
            window = input_ids.shape[1]
            encoder_embeds = []
            for i in range(window):
                ids = input_ids[:, i, :]
                mask = attention_mask[:, i, :] if attention_mask is not None else None

                output = self.embedding(ids)
                if self.embedding_size != self.hidden_size:
                    output = self.embedding_intermediate(output)

                for i in range(self.num_encoder_layers):
                    output = self.encoders[i](output, mask, training=training)

                encoder_embeds.append(output)

            encoder_embeds = tf.concat(encoder_embeds, axis=1)
            attention_mask = (
                tf.reshape(attention_mask, encoder_embeds.shape[:2])
                if attention_mask is not None
                else None
            )

            codes = tf.range(self.code_m, dtype=tf.int32)
            code_embeds = self.code_embedding(codes)

            encoder_output = self.dot_attention(
                code_embeds,
                encoder_embeds,
                encoder_embeds,
                mask=attention_mask,
            )
            attention_mask = tf.ones(encoder_output.shape[:-1])

        elif encoder_embed is not None:
            encoder_output = encoder_embed

        # decoder
        decoder_output = self.embedding(decoder_input_ids)
        if self.embedding_size != self.hidden_size:
            decoder_output = self.embedding_intermediate(decoder_output)

        decoder_mask = (
            None
            if decoder_attention_mask is None
            else self.create_look_ahead_mask(decoder_attention_mask)
        )

        for i in range(self.num_decoder_layers):
            decoder_output = self.decoders[i](
                decoder_output,
                encoder_output,
                decoder_mask,
                None,  # encoder output을 하나로 합치면서 의미가 없어짐
                training=training,
            )

        output = self.output_layer(decoder_output)

        if labels is not None:
            return (output, self.loss(labels, output))
        else:
            return (output,)

    def get_config(self):
        return {
            "embedding_size": self.embedding_size,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_encoder_layers": self.num_encoder_layers,
            "num_decoder_layers": self.num_decoder_layers,
            "vocab_size": self.vocab_size,
            "code_m": self.code_m,
            "pe": self.pe,
            "rate": self.rate,
        }

    def _get_sample_data(self):
        sample_data = {
            "input_ids": tf.random.uniform(
                (1, 1, 8), 0, self.vocab_size, dtype=tf.int64
            ),
            "decoder_input_ids": tf.random.uniform(
                (1, 1), 0, self.vocab_size, dtype=tf.int64
            ),
        }
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
