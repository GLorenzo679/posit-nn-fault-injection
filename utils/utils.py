import argparse
import csv
import os

import numpy as np
import softposit as sp
import tensorflow as tf
from models import convnet
from src import convnet_cifar10_inference
from tensorflow.keras.datasets import cifar10


def parse_args():
    """
    Parse the argument of the network
    """

    parser = argparse.ArgumentParser(
        description="Run a fault injection campaign with float/posit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--type",
        "-t",
        type=str,
        help="Numeric format to be used\n",
        required=True,
        choices=["posit8", "posit16", "posit32", "float32"],
    )
    parser.add_argument("--network-name", "-n", type=str, required=True, help="Network to be used", choices=["convnet"])
    parser.add_argument(
        "--data-set", "-d", type=str, required=True, help="Name of the dataset to use", choices=["CIFAR10"]
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Test set batch size")
    parser.add_argument("--size", "-s", type=int, default=512, help="Test set size")
    parser.add_argument("--force-n", type=int, default=None, help="Force n fault injections")
    parser.add_argument("--bit_len", "-b", type=int, required=True, help="Number of bits of data")
    parser.add_argument("--seed", type=int, default=0, help="Set seed for random values generation")

    parsed_args = parser.parse_args()

    return parsed_args


def get_network(network_name):
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :return: The loaded network
    """

    if network_name == "convnet":
        return convnet.model


def get_network_parameters(data_set, network_name, data_t):
    """
    Load the network with the specified name
    :param data_set: The name of the dataset
    :param network_name: The name of the network to load
    :param data_t: Data type
    :return: Network parameters such as: number of weights, number of layers, and tensor shape
    """

    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    path += "/data/" + data_set + "/" + network_name + "/"

    with tf.Session() as sess:
        # This object loads the model
        load_mod = tf.train.import_meta_graph(path + data_t + ".ckpt.meta")

        # Loading weights and biases and other stuff to the model
        load_mod.restore(sess, path + data_t + ".ckpt")

        w1 = sess.graph.get_tensor_by_name("Variable:0")

        print(tf.size(w1).eval())

        tensor_shape = w1.shape
        # we only consider convolutional layers for now
        num_layer = 2
        # for now manually computer (cv1, cv2, fc1, fc2, lin)
        num_weight_net = 4864 + 102464 + 1573248 + 73920 + 1930

    return num_weight_net, num_layer, tensor_shape


def get_loader(data_set):
    """
    Load the dataset
    :param data_set: The name of the dataset to load
    :return: Loader of the requested dataset
    """

    if data_set == "CIFAR10":
        return cifar10.load_data


def get_evaluator(network_name):
    """
    Get the network evaluator
    :param network_name: The name of the network to evaluate
    :return: Function used to evaluate the network
    """

    if network_name == "convnet":
        return convnet.evaluate


def get_sp_type(data_t):
    """
    Get the data type use for injection
    :param data_t: Data type
    :return: Selects the appropriate data type to use when corrupting weights
    """

    if data_t == "posit32":
        return sp.posit32
    elif data_t == "posit16":
        return sp.posit16
    elif data_t == "posit8":
        return sp.posit8
    elif data_t == "float32":
        return np.float32


def get_inference(data_set):
    """
    Get the function used to perform the inference step
    :param data_set: The name of the dataset
    :return: Function used to perform inference
    """

    # should also add a check on the network used
    if data_set == "CIFAR10":
        return convnet_cifar10_inference.Inference


def output_to_csv(results_path, fault, acc, golden_acc, top_5):
    """
    Output inference results to a csv file
    :param results_path: Path
    :param fault: Fault object
    :param acc: Accuracy for this fault
    :param golden_acc: Golden accuracy for this portion of the dataset (accuracy with no injection)
    :param top_5: Top_5 accuracy for this fault
    :return: None
    """

    with open(results_path, "a+") as file:
        headers = [
            "fault_id",
            "layer_index",
            "tensor_index",
            "bit_index",
            "accuracy",
            "golden_accuracy",
            "difference",
            "top_5",
            "weight_difference",
        ]

        writer = csv.DictWriter(file, delimiter=",", lineterminator="\n", fieldnames=headers)

        if fault.fault_id == 0:
            writer.writeheader()

        writer.writerow(
            {
                "fault_id": fault.fault_id,
                "layer_index": fault.layer_index,
                "tensor_index": fault.tensor_index,
                "bit_index": fault.bit_index,
                "accuracy": acc,
                "golden_accuracy": golden_acc,
                "difference": acc - golden_acc,
                "top_5": top_5,
                "weight_difference": fault.weight_start - fault.weight_corrupted,
            }
        )
