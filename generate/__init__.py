import sys
sys.path.append('src')
import logging
import json
import os
import numpy as np
import tensorflow as tf
import model, sample, encoder
import azure.functions as func
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model_name=os.environ['Model']
length=int(os.environ['Length'])
seed=None
nsamples=1
batch_size=1
temperature=1
top_k=0
top_p=0.0
if batch_size is None:
    batch_size = 1
assert nsamples % batch_size == 0

enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sessStart = tf.Session(graph=tf.Graph(), config=config)
sess = sessStart.__enter__()
context = tf.placeholder(tf.int32, [batch_size, None])
np.random.seed(seed)
tf.set_random_seed(seed)

output = sample.sample_sequence(
    hparams=hparams,
    length=length,
    context=context,
    batch_size=batch_size,
    temperature=temperature, top_k=top_k, top_p=top_p
)

saver = tf.train.Saver()
ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
saver.restore(sess, ckpt)
def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        req_body = req.get_json()
    except ValueError:
        pass
    else:
        raw_text = req_body.get('text')

    context_tokens = enc.encode(raw_text)
    generated = 0
    for _ in range(nsamples // batch_size):
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])
            logging.info("Input: " + raw_text)
            logging.info("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
            logging.info(text)
            return func.HttpResponse(text, status_code=200)