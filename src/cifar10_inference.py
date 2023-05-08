from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.contrib.layers import flatten
import sys
import csv
import os
import random
from Injection import Injection

np.random.seed(1)
tf.set_random_seed(2)


# Load CIFAR10 Dataset
data_set = "CIFAR-10"
print("Dataset is:", data_set)
(_, _), (X_test, y_test) = cifar10.load_data()
# somehow y_train comes as a 2D nx1 matrix
y_test = y_test.reshape(y_test.shape[0])

assert len(X_test) == len(y_test)

test_size = 512
#rand_index = random.randrange(0, len(X_test) - test_size)
rand_index = np.random.randint(0, len(X_test) - test_size)
X_test = X_test[rand_index:rand_index + test_size]
y_test = y_test[rand_index:rand_index + test_size]

print("\nImage Shape: {}\n".format(X_test[0].shape))
print("Test Set: {} samples".format(len(X_test)))

"""## Setup TensorFlow
The `EPOCH` and `BATCH_SIZE` values affect the training speed and model accuracy.
"""

BATCH_SIZE = 128
print("Batch size:", BATCH_SIZE)

# Set Posit data types
if len(sys.argv) > 1:
    data_t = sys.argv[1]
    if sys.argv[1] == "posit32":
        eps = 1e-8
        posit = np.posit32
        tf_type = tf.posit32
    elif sys.argv[1] == "posit16":
        eps = 1e-4
        posit = np.posit16
        tf_type = tf.posit16
    elif sys.argv[1] == "posit8":
        eps = 0.015625
        posit = np.posit8
        tf_type = tf.posit8
    elif sys.argv[1] == "float32":
        eps = 1e-8
        posit = np.float32
        tf_type = tf.float32
else:
    sys.exit("No data type provided in args")

# confirm dtype
print("\nType is:", data_t)

# Normalize data
X_test = ((X_test - 127.5) / 127.5).astype(posit)

print("Input data type: {}".format(type(X_test[0, 0, 0, 0])))

def Convnet(x, training):
    """## Implementation of CifarNet
    Implements the [CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)

    # Input
    The CifarNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since CIFAR10 images are RGB, C is 3 in this case.

    # Output
    Return the forwarded prediction - logits.
    """

    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x64.
    conv1_W = tf.Variable(
        tf.truncated_normal(shape=(5, 5, 3, 64), mean=mu, stddev=sigma, dtype=tf_type)
    )
    conv1_b = tf.Variable(tf.zeros(64, dtype=tf_type))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding="SAME") + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 32x32x64. Output = 16x16x64.
    conv1 = tf.nn.max_pool(
        conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )

    # Local Response Normalization
    # Use BatchNorm instead

    # Batch Normalization
    # conv1 = batch_norm(conv1, 64, training)
    # conv1 = tf.compat.v1.layers.BatchNormalization()(conv1)

    # Layer 2: Convolutional. Output = 16x16x64.
    conv2_W = tf.Variable(
        tf.truncated_normal(shape=(5, 5, 64, 64), mean=mu, stddev=sigma, dtype=tf_type)
    )
    conv2_b = tf.Variable(tf.zeros(64, dtype=tf_type))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding="SAME") + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Local Response Normalization
    # conv2 = tf.cast(conv2, tf.float32)
    # norm2 = tf.nn.lrn(conv2, 4, bias=posit(1.0), alpha=posit(0.001 / 9.0), beta=posit(0.75))
    # norm2 = tf.cast(norm2, tf_type)

    # Batch Normalization
    # conv2 = batch_norm(conv2, 64, training)

    # Pooling. Input = 16x16x64. Output = 8x8x64.
    conv2 = tf.nn.max_pool(
        conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID"
    )

    # Dropout
    # conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 8x8x64. Output = 4096.
    fc0 = flatten(conv2)
    dim = fc0.get_shape()[1].value

    # Layer 3: Fully Connected. Input = 4096. Output = 384.
    fc1_W = tf.Variable(
        tf.truncated_normal(shape=(dim, 384), mean=mu, stddev=sigma, dtype=tf_type)
    )
    fc1_b = tf.Variable(tf.zeros(384, dtype=tf_type))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    # fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 384. Output = 192.
    fc2_W = tf.Variable(
        tf.truncated_normal(shape=(384, 192), mean=mu, stddev=sigma, dtype=tf_type)
    )
    fc2_b = tf.Variable(tf.zeros(192, dtype=tf_type))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    # fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Linear layer(WX + b). Input = 192. Output = 10.
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    fc3_W = tf.Variable(
        tf.truncated_normal(shape=(192, 10), mean=mu, stddev=sigma, dtype=tf_type)
    )
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

    # out = tf.identity(logits+0)
    # return out


"""## Features and Labels

`x` is a placeholder for a batch of input images.
`y` is a placeholder for a batch of output labels.
"""

x = tf.placeholder(tf_type, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
training = tf.placeholder(tf.bool)

"""## Model Evaluation
Evaluate how well the loss and accuracy of the model for a given dataset.
"""

logits = Convnet(x, training)

correct_prediction = tf.equal(tf.argmax(logits, 1, output_type=tf.int32), y)
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
in_top5 = tf.nn.in_top_k(tf.cast(logits, tf.float32), y, k=5)
top5_operation = tf.reduce_mean(tf.cast(in_top5, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_top5 = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, BATCH_SIZE):
        print(f"\nBatch {int(offset/BATCH_SIZE + 1)}:")
        batch_x, batch_y = (
            X_data[offset : offset + BATCH_SIZE],
            y_data[offset : offset + BATCH_SIZE],
        )
        accuracy, top5 = sess.run(
            [accuracy_operation, top5_operation],
            feed_dict={x: batch_x, y: batch_y, training: False},
        )

        print(f"\tAccuracy: {accuracy}\tTop-5: {top5}")

        total_accuracy += accuracy * len(batch_x)
        total_top5 += top5 * len(batch_x)
    return (total_accuracy / num_examples, total_top5 / num_examples)


"""## Remove all training nodes
Only inference step will be computed with pretrained weights.
Therefore, training nodes are not neccessary at all.

If want to continue training the network (e.g. on transfer learning) comment this line.
"""
tf.graph_util.remove_training_nodes(tf.get_default_graph().as_graph_def())

path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
data_dir = path + "/data/CIFAR10/"
model_name = data_dir + sys.argv[1] + ".ckpt"

assert os.path.exists(data_dir), "The directory %s does not exist!" % data_dir


def modify_weight(weight, fault):
    mask = "0b" + "0"*(31 - fault.bit_index) + str(fault.bit_value) + "0"*(fault.bit_index)
    print(f"mask: {mask}")
    fault.setWeightStart(weight, mask)

    print(f"Start:\t {fault.weight_start}")
    print(f"End:\t {fault.weight_corrupted}")

    return fault.weight_corrupted


def inference(fault):
    """## Load and cast the Pre-trained Model before Evaluate
    https://stackoverflow.com/a/47077472

    Once you are completely satisfied with your model, evaluate the performance of the model on the test set.
    """

    #with tf.Session(config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)) as sess:
    with tf.Session() as sess:
        previous_variables = [var_name for var_name, _ in tf.contrib.framework.list_variables(model_name)]
        sess.run(tf.global_variables_initializer())

        for variable in tf.global_variables():
            if variable.op.name in previous_variables:
                var = tf.contrib.framework.load_variable(model_name, variable.op.name)
                if var.dtype == np.posit32:
                    tf.add_to_collection("assignOps", variable.assign(tf.cast(var, tf_type)))
                else:
                    print("Var. found of type ", var.dtype)
                    tf.add_to_collection("assignOps", variable.assign(var))
                    
        sess.run(tf.get_collection("assignOps"))

        if(fault != None):
            # get tensor of layer to manipulate 
            layer_name = "Variable" if fault.layer_index == 0 else "Variable_" + str(fault.layer_index * 2)
            weights = sess.graph.get_tensor_by_name(f"{layer_name}:0")
            np_weights = weights.eval()
            np_weights[fault.tensor_index] = modify_weight(np_weights[fault.tensor_index], fault)
            sess.run(tf.assign(weights, tf.convert_to_tensor(np_weights, dtype=np.posit32)))

        print("\nPre-Trained Parameters loaded and casted as type", tf_type)

        print("Computing Acc. (Top-1) & Top-5...")

        val_acc, test_top5 = evaluate(X_test, y_test)

        # reset injected weight
        if(fault != None):
            np_weights[fault.tensor_index] = fault.weight_start
            sess.run(tf.assign(weights, tf.convert_to_tensor(np_weights, dtype=np.posit32)))

        print("Accuracy:", val_acc)
        print("Top-5:", test_top5)

    return val_acc, test_top5


def main():
    injection = Injection()
    # num_layer limited to convolutional layers only for now
    injection.createInjectionList(num_weight_net=10, num_bit_representation=32, num_layer=2, num_batch=5, batch_height=5, batch_width=3, batch_features=64)

    # perform inference without injection
    golden_acc, top_5 = inference(None)

    for fault in injection.fault_list:
        print(f"\nFault: {fault.fault_id}")
        # perform inference with injection
        acc, top_5 = inference(fault)

        with open(path + "/results/CIFAR10/" + data_t + "_injection.csv", "a+") as file:
            headers = ["fault_id",
                       "layer_index",
                       "tensor_index",
                       "bit_index",
                       "accuracy",
                       "golden_accuracy",
                       "difference",
                       "top_5",
                       "weight_difference"]
            writer = csv.DictWriter(file, delimiter=",", lineterminator='\n', fieldnames=headers)

            if fault.fault_id == 0:
                writer.writeheader()

            writer.writerow({"fault_id": fault.fault_id,
                            "layer_index": fault.layer_index,
                            "tensor_index": fault.tensor_index,
                            "bit_index": fault.bit_index,
                            "accuracy": acc,
                            "golden_accuracy": golden_acc,
                            "difference": acc - golden_acc,
                            "top_5": top_5,
                            "weight_difference": fault.weight_start - fault.weight_corrupted})

main()