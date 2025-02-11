HW#5 from https://ufal.mff.cuni.cz/courses/npfl114/2223-summer#assignments

cnn_manual

To pass this assignment, you need to manually implement the forward and backward pass through a 2D convolutional layer. Start with the cnn_manual.py template, which constructs a series of 2D convolutional layers with ReLU activation and valid padding, specified in the args.cnn option. The args.cnn contains comma-separated layer specifications in the format filters-kernel_size-stride.

Of course, you cannot use any TensorFlow convolutional operation (instead, implement the forward and backward pass using matrix multiplication and other operations), nor the tf.GradientTape for gradient computation.

To make debugging easier, the template supports a --verify option, which allows comparing the forward pass and the three gradients you compute in the backward pass to correct values.


cags_classification

The goal of this assignment is to use a pretrained model, for example the EfficientNetV2-B0, to achieve best accuracy in CAGS classification.

The CAGS dataset consists of images of cats and dogs of size 224*224, each classified in one of the 34 breeds and each containing a mask indicating the presence of the animal. To load the dataset, use the cags_dataset.py module. The dataset is stored in a TFRecord file and each element is encoded as a tf.train.Example, which is decoded using the CAGS.parse method.

To load the EfficientNetV2-B0, use the tf.keras.applications.efficientnet_v2.EfficientNetV2B0 class, which constructs a Keras model, downloading the weights automatically. However, you can use any model from tf.keras.applications in this assignment.

An example performing classification of given images is available in image_classification.py.

A note on finetuning: each tf.keras.layers.Layer has a mutable trainable property indicating whether its variables should be updated – however, after changing it, you need to call .compile again (or otherwise make sure the list of trainable variables for the optimizer is updated). Furthermore, training argument passed to the invocation call decides whether the layer is executed in training regime (neurons gets dropped in dropout, batch normalization computes estimates on the batch) or in inference regime. There is one exception though – if trainable == False on a batch normalization layer, it runs in the inference regime even when training == True.


cags_segmentation

The goal of this assignment is to use a pretrained model, for example the EfficientNetV2-B0, to achieve best image segmentation IoU score on the CAGS dataset. The dataset and the EfficientNetV2-B0 is described in the cags_classification assignment. Nevertheless, you can again use any model from tf.keras.applications in this assignment.

A mask is evaluated using intersection over union (IoU) metric, which is the intersection of the gold and predicted mask divided by their union, and the whole test set score is the average of its masks' IoU. A TensorFlow compatible metric is implemented by the class MaskIoUMetric of the cags_dataset.py module, which can also evaluate your predictions (either by running with --task=segmentation --evaluate=path arguments, or using its evaluate_segmentation_file method).

