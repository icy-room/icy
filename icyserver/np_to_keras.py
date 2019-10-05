#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import backend as K

from transformers import TFGPT2LMHeadModel as GPT2Model, GPT2Tokenizer, GPT2Config



def map_name(name):
    name_map = [
        ('wpe/embeddings:0', 'wpe:0'),
        ('wte/weight:0', 'wte:0'),
        ('tfgp_t2lm_head_model/transformer/', 'model/'),
        ('h_._', 'h'),
        ('/weight:0', '/w:0'),
        ('/bias:0', '/b:0'),
        ('/gamma:0', '/g:0'),
        ('/beta:0', '/b:0'),
    ]
    for k, v in name_map:
        name = name.replace(k, v)
    return name


def main(base_config, npfile, outdir):
    config = GPT2Config.from_pretrained(base_config)
    tokenizer = GPT2Tokenizer.from_pretrained(base_config)
    model = GPT2Model(config)

    np_weights = np.load(npfile, allow_pickle=True).item()

    tf_inputs = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]])
    model(tf_inputs, training=False)

    symbolic_weights = model.trainable_weights + model.non_trainable_weights
    weight_value_tuples = []
    for w in symbolic_weights:
        name = w.name
        np_w = np_weights[map_name(name)]
        if name.endswith('weight:0') and 'wte' not in name:
            np_w = np_w[0]
        if name.endswith('bias:0'):
            np_w = np_w[None, :]
        weight_value_tuples.append((w, np_w))
    K.batch_set_value(weight_value_tuples)

    os.makedirs(outdir, exist_ok=True)
    config.save_pretrained(outdir)
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print('Saved')

if __name__ == '__main__':
    import fire
    fire.Fire(main)

