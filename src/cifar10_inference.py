from __future__ import division, print_function

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10


class Inference:
    def __init__(self, data_t, model, evaluator):
        np.random.seed(1)
        tf.set_random_seed(2)

        self.data_t = data_t
        self.model = model
        self.evaluator = evaluator

        # Load CIFAR10 Dataset
        data_set = "CIFAR-10"
        print("Dataset is:", data_set)
        (_, _), (X_test, self.y_test) = cifar10.load_data()
        # somehow y_train comes as a 2D nx1 matrix
        self.y_test = self.y_test.reshape(self.y_test.shape[0])

        assert len(X_test) == len(self.y_test)

        self.test_size = 128
        # rand_index = random.randrange(0, len(X_test) - test_size)
        rand_index = np.random.randint(0, len(X_test) - self.test_size)
        X_test = X_test[rand_index : rand_index + self.test_size]
        self.y_test = self.y_test[rand_index : rand_index + self.test_size]

        print("\nImage Shape: {}\n".format(X_test[0].shape))
        print("Test Set: {} samples".format(len(X_test)))

        """## Setup TensorFlow
        The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
        """

        self.BATCH_SIZE = 128
        print("Batch size:", self.BATCH_SIZE)

        # Set Posit data types
        if data_t == "posit32":
            self.posit = np.posit32
            self.tf_type = tf.posit32
        elif data_t == "posit16":
            self.posit = np.posit16
            self.tf_type = tf.posit16
        elif data_t == "posit8":
            self.posit = np.posit8
            self.tf_type = tf.posit8
        elif data_t == "float32":
            self.posit = np.float32
            self.tf_type = tf.float32
        else:
            sys.exit("Incorrect data type provided in args")

        # confirm dtype
        print("\nType is:", self.data_t)

        # Normalize data
        self.X_test = ((X_test - 127.5) / 127.5).astype(self.posit)

        print("Input data type: {}".format(type(self.X_test[0, 0, 0, 0])))

        """## Features and Labels

        `x` is a placeholder for a batch of input images.
        `y` is a placeholder for a batch of output labels.
        """

        self.x = tf.placeholder(self.tf_type, (None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.training = tf.placeholder(tf.bool)

        """## Model Evaluation
        Evaluate how well the loss and accuracy of the model for a given dataset.
        """

        logits = self.model(self.x, self.training, self.tf_type)

        correct_prediction = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), self.y)
        self.accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        in_top5 = tf.nn.in_top_k(tf.cast(logits, tf.float32), self.y, k=5)
        self.top5_operation = tf.reduce_mean(tf.cast(in_top5, tf.float32))

        """## Remove all training nodes
        Only inference step will be computed with pretrained weights.
        Therefore, training nodes are not neccessary at all.

        If want to continue training the network (e.g. on transfer learning) comment this line.
        """
        tf.graph_util.remove_training_nodes(tf.get_default_graph().as_graph_def())

        self.path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        self.data_dir = self.path + "/data/CIFAR10/"
        self.model_name = self.data_dir + data_t + ".ckpt"

        assert os.path.exists(self.data_dir), "The directory %s does not exist!" % self.data_dir

    def evaluate(self, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        total_top5 = 0
        sess = tf.get_default_session()

        for offset in range(0, num_examples, self.BATCH_SIZE):
            print(f"\nBatch {int(offset/self.BATCH_SIZE + 1)}:")
            batch_x, batch_y = (
                X_data[offset : offset + self.BATCH_SIZE],
                y_data[offset : offset + self.BATCH_SIZE],
            )
            accuracy, top5 = sess.run(
                [self.accuracy_operation, self.top5_operation],
                feed_dict={self.x: batch_x, self.y: batch_y, self.training: False},
            )

            print(f"\tAccuracy: {accuracy}\tTop-5: {top5}")

            total_accuracy += accuracy * len(batch_x)
            total_top5 += top5 * len(batch_x)
        return (total_accuracy / num_examples, total_top5 / num_examples)

    def compute_inference(self, fault=None):
        """## Load and cast the Pre-trained Model before Evaluate
        https://stackoverflow.com/a/47077472

        Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
        """

        # with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)) as sess:
        with tf.Session() as sess:
            previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(self.model_name)]
            sess.run(tf.global_variables_initializer())

            for variable in tf.global_variables():
                if variable.op.name in previous_variables:
                    var = tf.contrib.framework.load_variable(self.model_name, variable.op.name)
                    if var.dtype == np.posit32:
                        tf.add_to_collection("assignOps", variable.assign(tf.cast(var, self.tf_type)))
                    else:
                        print("Var. found of type ", var.dtype)
                        tf.add_to_collection("assignOps", variable.assign(var))

            sess.run(tf.get_collection("assignOps"))

            if fault != None:
                # get tensor of layer to manipulate
                layer_name = "Variable" if fault.layer_index == 0 else "Variable_" + str(fault.layer_index * 2)
                weights = sess.graph.get_tensor_by_name(f"{layer_name}:0")
                np_weights = weights.eval()
                fault.set_weight(np_weights[fault.tensor_index])
                np_weights[fault.tensor_index] = fault.weight_corrupted

                print(f"Start:\t {fault.weight_start}")
                print(f"End:\t {fault.weight_corrupted}")

                sess.run(tf.assign(weights, tf.convert_to_tensor(np_weights, dtype=np.posit32)))

            print("\nPre-Trained Parameters loaded and casted as type", self.tf_type)

            print("Computing Acc. (Top-1) & Top-5...")

            # val_acc, test_top5 = self.evaluator(self.X_test, self.y_test, self.BATCH_SIZE, self.x, self.y, self.training, self.accuracy_operation, self.top5_operation)

            val_acc, test_top5 = self.evaluate(self.X_test, self.y_test)

            # reset injected weight
            if fault != None:
                np_weights[fault.tensor_index] = fault.weight_start
                sess.run(tf.assign(weights, tf.convert_to_tensor(np_weights, dtype=np.posit32)))

            print("Accuracy:", val_acc)
            print("Top-5:", test_top5)

        return val_acc, test_top5
