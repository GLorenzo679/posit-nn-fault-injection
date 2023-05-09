import argparse
import csv

import numpy as np
import softposit as sp
from models import convnet
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
    parser.add_argument("--network-name", "-n", type=str, required=True, help="Network to be used", choices=["Convnet"])
    parser.add_argument(
        "--data-set", "-d", type=str, required=True, help="Name of the dataset to use", choices=["CIFAR10"]
    )
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Test set batch size")
    parser.add_argument("--size", "-s", type=int, default=512, help="Test set size")
    parser.add_argument("--force-n", type=int, default=None, help="Force n fault injections")

    parsed_args = parser.parse_args()

    return parsed_args


def get_network(network_name):
    """
    Load the network with the specified name
    :param network_name: The name of the network to load
    :param device: the device where to load the network
    :param root: the directory where to look for weights
    :return: The loaded network
    """

    if network_name == "Convnet":
        return convnet.model


def get_loader(data_set):
    if data_set == "CIFAR10":
        return cifar10.load_data


def get_evaluator(network_name):
    if network_name == "Convnet":
        return convnet.evaluate


def get_sp_type(data_t):
    if data_t == "posit32":
        return sp.posit32
    elif data_t == "posit16":
        return sp.posit16
    elif data_t == "posit8":
        return sp.posit8
    elif data_t == "float32":
        return np.float32


def output_to_csv(results_path, fault, acc, golden_acc, top_5):
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
