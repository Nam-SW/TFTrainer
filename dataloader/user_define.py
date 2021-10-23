import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# TODO: share_values 변수를 적어주지 않아도 공유해서 쓸 수 있게.


def _tokenize(sample, share_values):
    sample["tokenized"] = share_values.get("tokenizer")(sample["content"])["input_ids"]
    return sample


def _grouping(sample, share_values):
    def _padding(data):
        return pad_sequences(
            data,
            share_values.get("seq_len"),
            padding="post",
            truncating="post",
        )

    bos = [share_values.get("tokenizer").bos_token_id]
    eos = [share_values.get("tokenizer").eos_token_id]

    input_ids = []
    decoder_input_ids = []
    labels = []

    contents = [[] for _ in range(share_values.get("window") - 1)] + sample["tokenized"]
    talk_ids = [
        sample["talk_id"][0] for _ in range(share_values.get("window") - 1)
    ] + sample["talk_id"]

    s, e = 0, share_values.get("window")
    now_talk_id = talk_ids[0]

    while len(contents) > e:
        talk_id = talk_ids[e]

        if now_talk_id != talk_id:
            contents = [[] for _ in range(share_values.get("window") - 1)] + contents[
                s + share_values.get("window") :
            ]
            talk_ids = [
                talk_id for _ in range(share_values.get("window") - 1)
            ] + talk_ids[s + share_values.get("window") :]
            s, e = 0, share_values.get("window")
            now_talk_id = talk_id
            continue

        input_ids += contents[s:e]
        decoder_input_ids.append(bos + contents[e])
        labels.append(contents[e] + eos)

        s += 1
        e += 1

    return {
        "input_ids": _padding(input_ids).reshape(
            (-1, share_values.get("window"), share_values.get("seq_len")),
        ),
        "decoder_input_ids": _padding(decoder_input_ids),
        "labels": _padding(labels),
    }


def data_collator(x, y):
    x["attention_mask"] = tf.cast(tf.not_equal(x["input_ids"], 0), tf.int32)
    x["decoder_attention_mask"] = tf.cast(
        tf.not_equal(x["decoder_input_ids"], 0), tf.int32
    )
    return (x, y)
