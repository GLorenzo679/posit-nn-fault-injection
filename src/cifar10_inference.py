from __future__ import division, print_function

import os

import numpy as np
import tensorflow as tf


class Inference:
    def __init__(self, data_t, nn_model, evaluator, batch_size, data_set_size, data_set, loader):
        np.random.seed(1)
        tf.set_random_seed(2)

        self.evaluator = evaluator

        # load dataset
        print("Dataset is:", data_set)
        (_, _), (X_test, y_test) = loader()
        # somehow y_train comes as a 2D nx1 matrix
        self.y_test = y_test.reshape(y_test.shape[0])

        assert len(X_test) == len(self.y_test)

        # take a random subset of test samples of dimension = data_set_size
        rand_index = np.random.randint(0, len(X_test) - data_set_size)
        X_test = X_test[rand_index : rand_index + data_set_size]
        self.y_test = self.y_test[rand_index : rand_index + data_set_size]

        print("\nImage Shape: {}\n".format(X_test[0].shape))
        print("Test Set: {} samples".format(len(X_test)))

        # size of batches to process
        # num_batches =  data_set_size / batch_size
        self.batch_size = batch_size
        print("Batch size:", self.batch_size)

        # Set data type constructors to use
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

        # confirm data_t
        print("\nType is:", data_t)

        # Normalize data and convert it to selected data_t
        self.X_test = ((X_test - 127.5) / 127.5).astype(self.posit)

        print("Input data type: {}".format(type(self.X_test[0, 0, 0, 0])))

        """## Features and Labels

        `x` is a placeholder for a batch of input images.
        `y` is a placeholder for a batch of output labels.
        """
        self.x = tf.placeholder(self.tf_type, (None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.training = tf.placeholder(tf.bool)

        # MAYBE WE CAN MOVE THIS NEXT PART

        # setup operations to evaluate the model (prediction accuracy and top-5)
        logits = nn_model(self.x, self.tf_type)

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

        path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
        data_dir = path + "/data/" + data_set + "/"
        self.model_name = data_dir + data_t + ".ckpt"

        assert os.path.exists(data_dir), "The directory %s does not exist!" % data_dir

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
                    # this is not ok --> need to change in order to generalize data type
                    # if var.dtype == np.posit32:
                    if var.dtype == self.posit:
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

                sess.run(tf.assign(weights, tf.convert_to_tensor(np_weights, dtype=self.posit)))

            print("\nPre-Trained Parameters loaded and casted as type", self.tf_type)

            print("Computing Acc. (Top-1) & Top-5...")

            # perform evaluation on model
            val_acc, test_top5 = self.evaluator(
                self.X_test,
                self.y_test,
                self.batch_size,
                self.x,
                self.y,
                self.training,
                self.accuracy_operation,
                self.top5_operation,
            )

            # reset injected weight
            if fault != None:
                np_weights[fault.tensor_index] = fault.weight_start
                sess.run(tf.assign(weights, tf.convert_to_tensor(np_weights, dtype=self.posit)))

            print("Mean accuracy:", val_acc)
            print("Mean top-5:", test_top5)

        return val_acc, test_top5
