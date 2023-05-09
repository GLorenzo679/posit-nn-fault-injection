import csv
import os
import sys

import softposit as sp
from models.convnet import Convnet
from src.cifar10_inference import Inference
from src.Injection import Injection
from utils.evaluate import evaluate


def main():
    PATH = os.path.abspath(os.path.dirname(__file__))

    data_t = sys.argv[1]

    inference = Inference(data_t, Convnet, evaluate)

    injection = Injection()

    injection.createInjectionList(
        num_weight_net=10,
        num_bit_representation=32,
        num_layer=2,  # num_layer limited to convolutional layers only for now
        num_batch=5,
        batch_height=5,
        batch_width=3,
        batch_features=64,
        type=sp.posit32,
    )

    # change to .../results/{model_name provided by user}/ + data_t...
    results_path = PATH + "/results/CIFAR10/" + data_t + "_injection.csv"

    # perform inference without injection
    golden_acc, _ = inference.compute_inference()

    for fault in injection.fault_list:
        print(f"\nFault: {fault.fault_id}")
        # perform inference with injection
        acc, top_5 = inference.compute_inference(fault)

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


if __name__ == "__main__":
    main()
