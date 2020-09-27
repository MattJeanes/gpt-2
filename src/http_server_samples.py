#!/usr/bin/env python3

import encoder
import sample
import model
import fire
import json
import os
import sys
import gc
import numpy as np
import tensorflow as tf
import re
from threading import Lock
import flask
from flask import request, jsonify, make_response
tf.logging.set_verbosity(tf.logging.ERROR)

current_model_name = None
seed = None
nsamples = 1
batch_size = 1
length = int(os.environ.get('LENGTH', '50'))
temperature = 1
top_k = 0
top_p = 0.0
if batch_size is None:
    batch_size = 1
assert nsamples % batch_size == 0

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sessStart = None
sess = None
context = None
output = None
enc = None
lock = Lock()

app = flask.Flask(__name__)
# app.config["DEBUG"] = True


@app.route('/api/generate', methods=['POST'])
def home():
    with lock:
        raw_text = request.json.get('text', '')
        model_name = request.json.get('model', '')
        if len(raw_text) == 0:
            no_input = True
            raw_text = '<|endoftext|>'
        else:
            no_input = False
        if model_name != current_model_name:
            load_model(model_name)
        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            out = sess.run(output, feed_dict={
                context: [context_tokens for _ in range(batch_size)]
            })[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i])
                if no_input:
                    text = text.lstrip()
                print("Input: " + raw_text)
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
                return make_response(
                    jsonify(
                        {"text": text}
                    ),
                    200
                )


def load_model(model_name):
    global  sess, context, output, length, enc, current_model_name
    if not os.path.exists(os.path.join('models', model_name)):
        raise Exception("Invalid model")
    if sess is not None:
        sess.close()
        sess = None
        context = None
        output = None
        enc = None
        gc.collect()
    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx)
    sess = tf.Session(config=config)
    context = tf.placeholder(tf.int32, [batch_size, None])
    np.random.seed(seed)
    tf.set_random_seed(seed)
    output = sample.sample_sequence(
        hparams=hparams, length=length,
        context=context,
        batch_size=batch_size,
        temperature=temperature, top_k=top_k, top_p=top_p
    )

    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
    saver.restore(sess, ckpt)
    current_model_name = model_name


app.run(host='0.0.0.0')
