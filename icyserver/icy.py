#!/usr/bin/env python

import tensorflow as tf

from transformers import TFGPT2LMHeadModel as GPT2Model, GPT2Tokenizer


class PatchedTokenier(GPT2Tokenizer):
    def tokenize(self, text, **kwargs):
        """ Converts a string in a sequence of tokens (string), using the tokenizer.
            Split in words for word-based vocabulary or sub-words for sub-word-based
            vocabularies (BPE/SentencePieces/WordPieces).

            Take care of added tokens.
        """
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text:
                return []
            if not tok_list:
                return self._tokenize(text, **kwargs)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.added_tokens_encoder \
                            and sub_text not in self.all_special_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            return sum((self._tokenize(token, **kwargs) if token not \
                    in self.added_tokens_encoder and token not in self.all_special_tokens \
                    else [token] for token in tokenized_text), [])

        added_tokens = list(self.added_tokens_encoder.keys()) + self.all_special_tokens
        tokenized_text = split_on_tokens(added_tokens, text)
        return tokenized_text


def select(tensor, paths, *, axis=0):
    paths = paths[:, tf.newaxis]
    if isinstance(tensor, (list, tuple)):
        tensor = tf.stack(tensor, axis=0)
    if axis == 0:
        return tf.gather_nd(tensor, paths)
    if axis == 1:
        return tf.gather_nd(tensor, [paths] * tensor.shape[0], batch_dims=axis)
    raise NotImplementedError


def choose_top_1(accumu_probs, logits):
    softmaxes = tf.nn.log_softmax(logits)
    probs, indexes = tf.nn.top_k(softmaxes, k=1)
    accumu_probs = accumu_probs + probs[:, 0]
    return indexes[:, 0], accumu_probs


def top_k_beams(accumu_probs, logits, k):
    softmaxes = tf.nn.log_softmax(logits)
    accumu_probs = accumu_probs[:, tf.newaxis] + softmaxes
    flat_accumu_probs = tf.reshape(accumu_probs, [-1])
    accumu_probs, indexes = tf.nn.top_k(flat_accumu_probs, k=k)
    tokens = indexes % logits.shape[-1]
    paths = indexes // logits.shape[-1]
    return paths, tokens, accumu_probs


def get_past_shape(hparams, batch_size=None, sequence=None):
    return [hparams.n_layer, batch_size, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def new_icy(model_name):
    model = GPT2Model.from_pretrained(model_name)
    tokenizer = PatchedTokenier.from_pretrained(model_name)

    class Icy:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
            self.max_context_size = 200
            self.predict_len = 10
            self.beam_size = 8
            self.beam_steps = 2

        def predict(self, context):
            context_ids = self.tokenizer.encode(context)
            if len(context_ids) <= 1:
                return 0, []
            context_ids = context_ids[-self.max_context_size:]
            print('The last 2 tokens are: {}'.format(self.tokenizer.convert_ids_to_tokens(context_ids[-2:])))

            # the last token may incomplete, we need to estimate it
            tokens, probs, past = self.estimate_first(context_ids)
            if len(tokens) == 0:
                return 0, []

            past = tf.stack(past, axis=0)
            past = select(past, tf.zeros(len(tokens), dtype=tf.int32), axis=1)
            tokens = tf.constant(tokens, dtype=tf.int32)
            tf_context_ids = tf.constant(context_ids[:-1], dtype=tf.int32)[tf.newaxis, :]
            tf_context_ids = tf.tile(tf_context_ids, [len(tokens), 1])
            tf_context_ids = tf.concat([tf_context_ids, tokens[:, tf.newaxis]], axis=-1)
            y = self._predict(tf_context_ids, past, tf.constant(probs))
            last_token_len = len(self.tokenizer.convert_ids_to_tokens(context_ids[-1]))
            return last_token_len, [self.tokenizer.decode(i) for i in y.numpy()[:, -self.predict_len-1:]]

        @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.int32)])
        def get_top_k(self, context_ids):
            context_ids = context_ids[None, :]
            logits, past = self.model(context_ids, past=None)
            logits = logits[:, -1]
            probs = tf.nn.log_softmax(logits)
            p, ix = tf.nn.top_k(probs, k=10000)  # TODO: move to param
            return p, ix, past

        def estimate_first(self, context_ids):
            last_token = self.tokenizer.convert_ids_to_tokens(context_ids[-1])
            context_ids = tf.constant(context_ids[:-1], dtype=tf.int32)
            probs, indexes, past = self.get_top_k(context_ids)
            probs = probs.numpy().tolist()
            indexes = indexes.numpy().tolist()
            accumu_probs = []
            candidates = []
            for tk_id, p in zip(indexes[0], probs[0]):
                token = self.tokenizer.convert_ids_to_tokens(tk_id)
                if token.startswith(last_token):
                    candidates.append(tk_id)
                    accumu_probs.append(p)
                    if len(candidates) >= self.beam_size:
                        break
            return candidates, accumu_probs, past

        @tf.function(input_signature=[
                                    tf.TensorSpec(shape=[None, None], dtype=tf.int32),  # context
                                    tf.TensorSpec(shape=get_past_shape(model.config), dtype=tf.float32),  # context
                                    tf.TensorSpec(shape=[None], dtype=tf.float32),  # context
                                    ])
        def _predict(self, context_ids, past, accumu_probs):
            past = tf.unstack(past)

            for _ in tf.range(self.beam_steps):
                logits, past = self.model(context_ids[:, -1:], past=past)
                logits = logits[:, -1]
                paths, tokens, accumu_probs = top_k_beams(accumu_probs, logits, self.beam_size)
                context_ids = select(context_ids, paths)
                context_ids = tf.concat([context_ids, tokens[:, tf.newaxis]], axis=-1)
                past = tf.unstack(select(past, paths, axis=1))

            past = tuple(past)
            for _ in tf.range(self.beam_steps, self.predict_len):
                logits, past = self.model(context_ids[:, -1:], past=past)
                logits = logits[:, -1]
                tokens, accumu_probs = choose_top_1(accumu_probs, logits)
                context_ids = tf.concat([context_ids, tokens[:, tf.newaxis]], axis=-1)
                #self.print_tokens(context_ids)

            return context_ids

        def print_tokens(self, context_ids):
            print('==========')
            for j, line in enumerate(context_ids):
                print(f'--{j}--')
                print(list(line.numpy()))
                print(self.tokenizer.decode(line.numpy()))

    return Icy(model, tokenizer)


if __name__ == '__main__':
    import time

    icy = new_icy('gpt2-medium')
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

