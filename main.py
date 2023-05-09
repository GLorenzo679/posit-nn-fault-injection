import os

from src.cifar10_inference import Inference
from src.Injection import Injection
from utils.utils import (
    get_evaluator,
    get_loader,
    get_network,
    get_sp_type,
    output_to_csv,
    parse_args,
)


def main(args):
    data_t = args.type
    network_name = args.network_name

    # initialize inference class
    inference = Inference(
        data_t,
        get_network(network_name),
        get_evaluator(network_name),
        args.batch_size,
        args.size,
        args.data_set,
        get_loader(args.data_set),
    )

    # create injection list
    injection = Injection()
    injection.create_injection_list(
        num_weight_net=10,
        num_bit_representation=32,
        num_layer=2,  # num_layer limited to convolutional layers only for now
        num_batch=5,
        batch_height=5,
        batch_width=3,
        batch_features=64,
        type=get_sp_type(data_t),
    )

    # setup path for results file
    PATH = os.path.abspath(os.path.dirname(__file__))
    results_path = PATH + "/results/" + args.data_set + "/" + data_t + "_injection.csv"

    # perform inference without injection
    golden_acc, _ = inference.compute_inference()

    # perform inference for every fault in fault list
    for fault in injection.fault_list:
        print(f"\nFault: {fault.fault_id}")
        # perform inference with injection
        acc, top_5 = inference.compute_inference(fault)
        # output results to csv in results_path
        output_to_csv(results_path, fault, acc, golden_acc, top_5)


if __name__ == "__main__":
    main(args=parse_args())
