"""Small CNN, Vision Transformer and U-Net for 64 x 64 multi-channel FMI tiles · TensorFlow / Keras · v1.1

Sized to train in seconds on a laptop CPU. Production work would fine-tune pre-trained backbones (ResNet,
EfficientNet, ViT); the ideas are the same. Arrays from fmi_core are (channels, rows, columns); every function
here accepts that layout and converts to Keras' channels-last internally.

Models are built lazily: TinyCNN(circular) describes the network, train_classifier builds it after fixing the
random seed, so every training is reproducible.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import warnings

import numpy as np
import tensorflow as tf
import keras

tf.get_logger().setLevel("ERROR")                                   # silence harmless retracing notices
warnings.filterwarnings("ignore", message=".*shuffle=True.*tf.data.Dataset.*")
from keras import layers
from sklearn.metrics import average_precision_score


def _cl(X):
    """(N, C, H, W) or (C, H, W) -> channels last."""
    X = np.asarray(X, np.float32)
    return np.moveaxis(X, -3, -1)


@keras.saving.register_keras_serializable(package="fmi_case")
class AzPad(layers.Layer):
    """Pad azimuth (width) circularly or with zeros; depth (height) always with zeros."""
    def __init__(self, circular=True, p=1, **kw):
        super().__init__(**kw); self.circular, self.p = circular, p

    def call(self, x):
        p = self.p
        if self.circular:
            x = tf.concat([x[:, :, -p:], x, x[:, :, :p]], axis=2)
        else:
            x = tf.pad(x, [[0, 0], [0, 0], [p, p], [0, 0]])
        return tf.pad(x, [[0, 0], [p, p], [0, 0], [0, 0]])

    def get_config(self):
        return {**super().get_config(), "circular": self.circular, "p": self.p}


def _conv(x, f, circular, name):
    x = AzPad(circular, name=f"{name}_pad")(x)
    x = layers.Conv2D(f, 3, padding="valid", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


class TinyCNN:
    """Three blocks of two 3 x 3 convolutions (8, 16, 32 filters), max pooling between blocks, global average pooling,
    one logit: does the tile hold a natural fracture?"""
    kind = "classifier"

    def __init__(self, circular=True):
        self.circular = circular

    def build(self, in_ch):
        inp = keras.Input((64, 64, in_ch))
        x = inp
        for b, f in enumerate((8, 16, 32)):
            x = _conv(x, f, self.circular, f"b{b + 1}a")
            x = _conv(x, f, self.circular, f"b{b + 1}b")
            if b < 2:
                x = layers.MaxPooling2D(2, name=f"pool{b + 1}")(x)
        x = layers.Activation("linear", name="feats")(x)          # 16 x 16 maps used by Grad-CAM
        out = layers.Dense(1, name="logit")(layers.GlobalAveragePooling2D()(x))
        return keras.Model(inp, out, name="tiny_cnn")


@keras.saving.register_keras_serializable(package="fmi_case")
class ClassToken(layers.Layer):
    def build(self, shape):
        self.cls = self.add_weight(shape=(1, 1, shape[-1]), initializer="zeros", name="cls")
        self.pos = self.add_weight(shape=(1, shape[1] + 1, shape[-1]), initializer=keras.initializers.RandomNormal(stddev=0.02), name="pos")

    def call(self, t):
        c = tf.repeat(self.cls, tf.shape(t)[0], axis=0)
        return tf.concat([c, t], 1) + self.pos


class TinyViT:
    """8 x 8 patches -> 64 tokens plus a class token, two transformer blocks (64 dims, 4 heads)."""
    kind = "classifier"

    def __init__(self, d=64, heads=4, depth=2, patch=8):
        self.d, self.heads, self.depth, self.patch = d, heads, depth, patch

    def build(self, in_ch):
        inp = keras.Input((64, 64, in_ch))
        t = layers.Conv2D(self.d, self.patch, strides=self.patch, name="embed")(inp)
        t = layers.Reshape((-1, self.d))(t)
        t = ClassToken(name="cls_pos")(t)
        scores = None
        for b in range(self.depth):
            h = layers.LayerNormalization(name=f"ln{b}a")(t)
            a, scores = layers.MultiHeadAttention(self.heads, self.d // self.heads, name=f"att{b}")(h, h, return_attention_scores=True)
            t = layers.Add()([t, a])
            h = layers.LayerNormalization(name=f"ln{b}b")(t)
            h = layers.Dense(2 * self.d, activation="gelu", name=f"mlp{b}a")(h)
            t = layers.Add()([t, layers.Dense(self.d, name=f"mlp{b}b")(h)])
        cls = layers.LayerNormalization(name="norm")(t[:, 0])
        out = layers.Dense(1, name="logit")(cls)
        m = keras.Model(inp, out, name="tiny_vit")
        m.attention_model = keras.Model(inp, scores)              # last block: (batch, heads, 65, 65)
        return m


class MiniUNet:
    """Encoder-decoder with skip links; 5 classes: background, open fracture, closed fracture, vug, stylolite."""
    kind = "segmenter"

    def __init__(self, circular=True, n_out=5):
        self.circular, self.n_out = circular, n_out

    def build(self, in_ch):
        inp = keras.Input((None, 64, in_ch))
        c = self.circular
        e1 = _conv(_conv(inp, 8, c, "e1a"), 8, c, "e1b")
        e2 = _conv(_conv(layers.MaxPooling2D(2)(e1), 16, c, "e2a"), 16, c, "e2b")
        e3 = _conv(_conv(layers.MaxPooling2D(2)(e2), 32, c, "e3a"), 32, c, "e3b")
        d2 = layers.Concatenate()([layers.UpSampling2D(2)(e3), e2])
        d2 = _conv(_conv(d2, 16, c, "d2a"), 16, c, "d2b")
        d1 = layers.Concatenate()([layers.UpSampling2D(2)(d2), e1])
        d1 = _conv(_conv(d1, 8, c, "d1a"), 8, c, "d1b")
        out = layers.Conv2D(self.n_out, 1, name="logits")(d1)
        return keras.Model(inp, out, name="mini_unet")


# ----------------------------------------------------------------------------- training
def _roll(x, *rest):
    s = tf.random.uniform([], 0, tf.shape(x)[1], dtype=tf.int32)          # single example: (H, W, C)
    return (tf.roll(x, s, axis=1),) + tuple(tf.roll(r, s, axis=1) if len(r.shape) >= 2 else r for r in rest)


class _Progress(keras.callbacks.Callback):
    def __init__(self, epochs, progress, Xv=None, yv=None, hist=None):
        super().__init__(); self.epochs, self.progress, self.Xv, self.yv, self.hist = epochs, progress, Xv, yv, hist
        self.t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        rec = {"epoch": epoch + 1, "train_loss": float(logs["loss"])}
        if self.Xv is not None:
            p = 1 / (1 + np.exp(-self.model.predict(self.Xv, batch_size=256, verbose=0)[:, 0]))
            rec["blind_AP"] = average_precision_score(self.yv, p) if self.yv.sum() else np.nan
        self.hist.append(rec)
        if self.progress:
            self.progress((epoch + 1) / self.epochs, f"Epoch {epoch + 1}/{self.epochs} · {time.time() - self.t0:.0f} s")


def train_classifier(spec, X, y, Xv=None, yv=None, epochs=8, lr=2e-3, bs=64, roll_aug=True, seed=0, progress=None):
    """Weighted binary cross-entropy (positive tiles up-weighted to balance the classes), Adam, optional azimuth roll."""
    keras.utils.set_random_seed(seed)
    Xc = _cl(X)
    model = spec.build(Xc.shape[-1])
    pos_w = (1 - y.mean()) / max(y.mean(), 1e-3)
    sw = np.where(y == 1, pos_w, 1.0).astype(np.float32)
    ds = tf.data.Dataset.from_tensor_slices((Xc, y.astype(np.float32)[:, None], sw)).shuffle(len(y), seed=seed)
    if roll_aug:
        ds = ds.map(_roll)
    ds = ds.batch(bs)
    model.compile(keras.optimizers.Adam(lr), keras.losses.BinaryCrossentropy(from_logits=True))
    hist = []
    model.fit(ds, epochs=epochs, shuffle=False, verbose=0, callbacks=[_Progress(epochs, progress, None if Xv is None else _cl(Xv), yv, hist)])
    return model, hist


def predict(model, X, bs=256):
    return 1 / (1 + np.exp(-model.predict(_cl(X), batch_size=bs, verbose=0)[:, 0]))


def train_unet(spec, X, T, epochs=8, lr=2e-3, bs=32, roll_aug=True, seed=0, progress=None, class_w=(1.0, 8.0, 12.0, 4.0, 4.0)):
    """Pixel-weighted cross-entropy: rare fracture and vug pixels count more than background."""
    keras.utils.set_random_seed(seed)
    Xc = _cl(X)
    model = spec.build(Xc.shape[-1])
    W = np.asarray(class_w, np.float32)[T]
    ds = tf.data.Dataset.from_tensor_slices((Xc, T.astype(np.int32), W)).shuffle(len(Xc), seed=seed)
    if roll_aug:
        ds = ds.map(_roll)
    ds = ds.batch(bs)
    model.compile(keras.optimizers.Adam(lr), keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    hist = []
    model.fit(ds, epochs=epochs, shuffle=False, verbose=0, callbacks=[_Progress(epochs, progress, hist=hist)])
    return model, hist


def segment_well(model, prep):
    """Run the fully convolutional U-Net over a whole well (height padded to a multiple of 4).
    Returns class probabilities (n_classes, H, W)."""
    H = prep.shape[1]
    pad = (-H) % 4
    x = _cl(np.pad(prep, ((0, 0), (0, pad), (0, 0))))[None]
    p = tf.nn.softmax(model(x, training=False), -1)[0, :H].numpy()
    return np.moveaxis(p, -1, 0)


# ----------------------------------------------------------------------------- interpretability
def grad_cam(model, x):
    """Grad-CAM for the fracture logit. x: (C, 64, 64). Returns a 64 x 64 map in [0, 1] and the probability."""
    sub = keras.Model(model.inputs, [model.get_layer("feats").output, model.output])
    xb = tf.convert_to_tensor(_cl(x)[None])
    with tf.GradientTape() as tape:
        f, logit = sub(xb, training=False)
        tape.watch(f)
        score = logit[:, 0]
    g = tape.gradient(score, f)
    w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * f, -1))[0]
    cam = tf.image.resize(cam[..., None], (64, 64))[..., 0].numpy()
    return cam / (cam.max() + 1e-8), float(1 / (1 + np.exp(-float(logit[0, 0]))))


def vit_attention(model, x):
    """Attention from the class token to the 64 patches in the last block, averaged over heads, as a 64 x 64 map."""
    xb = _cl(x)[None]
    p = float(predict(model, x[None])[0])
    a = model.attention_model(xb, training=False).numpy()[0].mean(0)[0, 1:].reshape(8, 8)
    a = np.kron(a, np.ones((8, 8)))
    return a / (a.max() + 1e-8), p
