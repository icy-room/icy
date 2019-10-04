#!/usr/bin/env python

import tensorflow as tf
from transformers import TFGPT2LMHeadModel as GPT2Model, GPT2Tokenizer

import tensorflow as tf


def select(tensor, paths):
    if isinstance(tensor, (tuple, list)):
        return [select(t, paths) for t in tensor]
    return tf.gather_nd(tensor, paths[:, tf.newaxis])


def top_k_beams(accumu_probs, logits, k):
    softmaxes = tf.nn.log_softmax(logits)
    accumu_probs = accumu_probs[:, tf.newaxis] + softmaxes
    flat_accumu_probs = tf.reshape(accumu_probs, [-1])
    accumu_probs, indexes = tf.nn.top_k(flat_accumu_probs, k=k)
    tokens = indexes % logits.shape[-1]
    paths = indexes // logits.shape[-1]
    return paths, tokens, accumu_probs


class Icy:
    def __init__(self, model_name):
        self.model = GPT2Model.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.max_context_size = 200
        self.predict_len = 5
        self.beam_size = 8

    def predict(self, context):
        context_ids = self.tokenizer.encode(context)
        x = tf.constant(context_ids[-self.max_context_size:-1], dtype=tf.int32)
        y = self._predict(x)
        last_token_len = len(self.tokenizer.decode(context_ids[-1:]))
        return last_token_len, [self.tokenizer.decode(i) for i in y.numpy()[:, -self.predict_len:]]

    @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32)])
    def _predict(self, context_ids):
        context_ids = context_ids[None, :]

        context_ids, past, accumu_probs = self.step(
                                    context_ids,
                                    None,
                                    tf.zeros(1, dtype=tf.float32))

        for i in tf.range(self.predict_len-1):
            context_ids, past, accumu_probs = self.step(context_ids, past, accumu_probs)

        return context_ids

    def step(self, context_ids, past, accumu_probs):
        if past is None:
            tiled = tf.concat([context_ids]*self.beam_size, axis=0)
            logits, past = self.model(tiled)
            paths, tokens, accumu_probs = top_k_beams(
                                            accumu_probs[:1],
                                            logits[:1,-1],
                                            self.beam_size)
        else:
            logits, past = self.model(context_ids[:, -1:], past=past)
            paths, tokens, accumu_probs = top_k_beams(
                                            accumu_probs,
                                            logits[:, -1],
                                            self.beam_size)
        context_ids = select(context_ids, paths)
        context_ids = tf.concat([context_ids, tokens[:, tf.newaxis]], axis=-1)

        #self.print_tokens(context_ids)

        return (context_ids,
                select(past, paths),
                accumu_probs)

    def print_tokens(self, context_ids):
        print('==========')
        for j, line in enumerate(context_ids):
            print(f'--{j}--')
            print(list(line.numpy()))
            print(self.tokenizer.decode(line.numpy()))


if __name__ == '__main__':
    import time

    icy = Icy('gpt2-medium')
    icy.predict('x a')

    t0 = time.time()
    n, candidates = icy.predict('''class Cat:
name: str
age: int

def __init__(self, name, age):
    self.name = name
    self.age = age

def __str__(self):
    return "Cat<name=''')

    print('t:{}'.format(time.time() - t0))
    for t in candidates:
        print('----')
        print(t)

