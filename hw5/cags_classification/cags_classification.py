#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

#    Define reasonable defaults and optionally more parameters.
# Also, you can set the number of threads to 0 to use all your CPU cores.
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=20, type=int, help="Batch size.")
parser.add_argument("--debug", default=False, action="store_true", help="If given, run functions eagerly.")
parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")


def main(args: argparse.Namespace) -> None:
    # Set the random seed and the number of threads.
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    if args.debug:
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    # Create logdir name
    args.logdir = os.path.join("logs", "{}-{}-{}".format(
        os.path.basename(globals().get("__file__", "notebook")),
        datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
        ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(args).items())))
    ))

    # Load the data. Note that both the "image" and the "mask" images
    # are represented using `tf.uint8`s in [0-255] range.
    cags = CAGS()

    # len(cags.train)) 2142б len(cags.dev) 306, len(cags.test)) 612

    # prepare data
    train = cags.train.map(lambda example: (example["image"], example["label"]))
    train = train.shuffle(5000, seed=args.seed).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    dev = cags.dev.map(lambda example: (example["image"], example["label"]))
    dev = dev.shuffle(5000, seed=args.seed).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    test = cags.test.map(lambda example: (example["image"])).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # print(list(train)[0])

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False, pooling="avg")
    backbone.trainable = False

    # model architecture
    inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
    hidden = backbone(inputs=inputs)
    hidden = tf.keras.layers.Flatten(name="flatten")(hidden)
    # add dense layer
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    outputs = tf.keras.layers.Dense(units=len(CAGS.LABELS), activation=tf.nn.softmax)(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()

    # compile and train
    model.compile(
        optimizer=tf.optimizers.Adam(jit_compile=False),
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    model.fit(train, epochs=args.epochs, validation_data=dev, callbacks=[tb_callback])

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_classification.txt"), "w", encoding="utf-8") as predictions_file:
        #    Predict the probabilities on the test set
        test_probabilities = model.predict(test, batch_size=args.batch_size)

        for probs in test_probabilities:
            print(np.argmax(probs), file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
