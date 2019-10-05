#!/usr/bin/env python3
import os
import sys
import fire
import json
import numpy as np
import tensorflow as tf


def main(mdir, output):
    """
    Convert openai gpt2 checkpoint to numpy format
    :mdir : String, which model to use
    """

    import model

    batch_size = 1

    hparams = model.default_hparams()
    with open(os.path.join(mdir, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])

        model.model(hparams=hparams, X=context, past=None, reuse=tf.AUTO_REUSE)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(mdir)
        saver.restore(sess, ckpt)

        variables = tf.all_variables()
        names = [v.name for v in variables]
        values = sess.run(variables)
        dic = dict(zip(names, values))
        np.save(output, dic)
        print('Saved')


if __name__ == '__main__':
    sys.path.append('src')
    fire.Fire(main)
