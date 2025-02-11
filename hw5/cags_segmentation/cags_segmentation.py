#!/usr/bin/env python3
import argparse
import datetime
import os
import re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # Report only TF errors by default

import numpy as np
import tensorflow as tf

from cags_dataset import CAGS

parser = argparse.ArgumentParser()
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--threads", default=0, type=int, help="Maximum number of threads to use.")
parser.add_argument("--experiment_name", default="cags_segmentation", type=str, help="Add to logdir name.")

parser.add_argument("--batch_size", default=250, type=int, help="Batch size.")
parser.add_argument("--epochs", default=1, type=int, help="Number of epochs.")

parser.add_argument("--start_neurons", default=8, type=int, help="Start number of neurons in conv layer.")
parser.add_argument("--activation", default="relu", type=str, help="Activation function.")
parser.add_argument("--dropout", default=0.5, type=float, help="Dropout rate.")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate for optimizer.")


def main(args: argparse.Namespace) -> None:
    # Fix random seeds and threads
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)

    # Create logdir name
    args.logdir = os.path.join("logs",
                               args.experiment_name,
                               "{}-{}-{}".format(
                                   os.path.basename(globals().get("__file__", "notebook")),
                                   datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
                                   ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in
                                             sorted(vars(args).items())))
                               ))

    # Load the data. Note that both the "image" and the "mask" images
    # are represented using `tf.uint8`s in [0-255] range.
    cags = CAGS()

    # prepare data
    train = cags.train.map(lambda example: (example["image"], example["mask"]))
    train = train.shuffle(5000, seed=args.seed)
    # if args.layers_aug: train = train.map(layers_augmentation)
    # if args.image_aug: train = train.map(image_augmentation)
    train = train.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    dev = cags.dev.map(lambda example: (example["image"], example["mask"]))
    dev = dev.shuffle(5000, seed=args.seed).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    test = cags.test.map(lambda example: (example["image"])).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Load the EfficientNetV2-B0 model. It assumes the input images are
    # represented in [0-255] range using either `tf.uint8` or `tf.float32` type.
    backbone = tf.keras.applications.EfficientNetV2B0(include_top=False)
    backbone.trainable = False

    # Extract features of different resolution. Assuming 224x224 input images
    # (you can set this explicitly via `input_shape` of the above constructor),
    # the below model returns five outputs with resolution 7x7, 14x14, 28x28, 56x56, 112x112.
    backbone = tf.keras.Model(
        inputs=backbone.input,
        outputs=[backbone.get_layer(layer).output for layer in [
             "top_activation", "block5e_add", "block3b_add", "block2b_add", "block1a_project_activation"]]
    )

    # for output in backbone.outputs:
    #     print(output)

    # model architecture
    inputs = tf.keras.layers.Input(shape=[CAGS.H, CAGS.W, CAGS.C])
    backbone_outputs = backbone(inputs=inputs)

    conv = tf.keras.layers.Conv2D(args.start_neurons * 16, kernel_size=3, padding="same")(backbone_outputs[0])
    conv = tf.keras.layers.Activation(args.activation)(conv)
    conv = tf.keras.layers.Conv2D(args.start_neurons * 16, kernel_size=3, padding="same")(conv)
    conv = tf.keras.layers.Activation(args.activation)(conv)

    mult = 16
    for i, b_output in enumerate(backbone_outputs[1:]): # 0, 1->14 (8 mult), 2->28 (4 mult), 3->56 (2 mult), 4->112 (1 mult)
        deconv = tf.keras.layers.Conv2DTranspose(args.start_neurons * mult/2, kernel_size=3, strides=2, padding="same")(conv)
        uconv = tf.keras.layers.concatenate([deconv, b_output])
        conv = tf.keras.layers.Dropout(args.dropout)(uconv)
        conv = tf.keras.layers.Conv2D(args.start_neurons * mult/2, kernel_size=3, padding="same")(conv)
        conv = tf.keras.layers.Activation(args.activation)(conv)
        conv = tf.keras.layers.Conv2D(args.start_neurons * mult/2, kernel_size=3, padding="same")(conv)
        conv = tf.keras.layers.Activation(args.activation)(conv)
        mult /= 2

    output = tf.keras.layers.Conv2DTranspose(1, kernel_size=3, strides=2, padding="same", activation="sigmoid")(conv)

    model = tf.keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="accuracy"), CAGS.MaskIoUMetric(name="IoU")],
    )
    tb_callback = tf.keras.callbacks.TensorBoard(args.logdir)

    model.summary()

    model.fit(
        x=train,
        epochs=args.epochs,
        validation_data=dev,
        callbacks=[tb_callback]
    )

    # Generate test set annotations, but in `args.logdir` to allow parallel execution.
    os.makedirs(args.logdir, exist_ok=True)
    with open(os.path.join(args.logdir, "cags_segmentation.txt"), "w", encoding="utf-8") as predictions_file:
        test_masks = model.predict(test)

        for mask in test_masks:
            zeros, ones, runs = 0, 0, []
            for pixel in np.reshape(mask >= 0.5, [-1]):
                if pixel:
                    if zeros or (not zeros and not ones):
                        runs.append(zeros)
                        zeros = 0
                    ones += 1
                else:
                    if ones:
                        runs.append(ones)
                        ones = 0
                    zeros += 1
            runs.append(zeros + ones)
            print(*runs, file=predictions_file)


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
