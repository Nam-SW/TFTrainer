import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# TODO: share_values 변수를 적어주지 않아도 공유해서 쓸 수 있게.


def _tokenize(sample, share_values):
    data = dict()

    def _padding(l):
        return pad_sequences(
            l,
            share_values.get("seq_len"),
            padding="post",
            truncating="post",
        )

    # emcode
    data["input_ids"] = [
        [share_values.get("tokenizer")[c] for c in q] for q in sample["Q"]
    ]
    data["decoder_input_ids"] = [
        [1] + [share_values.get("tokenizer")[c] for c in q] for q in sample["Q"]
    ]
    data["labels"] = [
        [share_values.get("tokenizer")[c] for c in q] + [2] for q in sample["Q"]
    ]

    # padding
    data["input_ids"] = _padding(data["input_ids"])
    data["decoder_input_ids"] = _padding(data["decoder_input_ids"])
    data["labels"] = _padding(data["labels"])

    return data


def data_collator(x, y):
    x["attention_mask"] = tf.cast(tf.not_equal(x["input_ids"], 0), tf.int32)
    x["decoder_attention_mask"] = tf.cast(
        tf.not_equal(x["decoder_input_ids"], 0), tf.int32
    )
    return (x, y)


aplly_list = [_tokenize]
