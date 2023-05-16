import os

from src.Injection import Injection
from utils.utils import (
    get_evaluator,
    get_inference,
    get_loader,
    get_network,
    get_network_parameters,
    get_sp_type,
    output_to_csv,
    parse_args,
)


def main(args):
    data_t = args.type
    network_name = args.network_name
    data_set = args.data_set
    inference_class = get_inference(data_set)

    # initialize inference class
    inference = inference_class(
        data_t,
        network_name,
        get_network(network_name),
        get_evaluator(network_name),
        args.batch_size,
        args.size,
        data_set,
        get_loader(args.data_set),
        args.seed,
    )

    num_weight_net, num_layer, tensor_shape = get_network_parameters(data_set, network_name, data_t)

    # create injection list
    injection = Injection()
    injection.create_injection_list(
        num_weight_net,
        num_layer,  # num_layer limited to convolutional layers only for now
        tensor_shape,
        num_bit_representation=args.bit_len,
        type=get_sp_type(data_t),
        number_of_faults=args.force_n,
    )

    # setup path for results file
    PATH = os.path.abspath(os.path.dirname(__file__))
    results_path = PATH + "/res/" + data_set + "/" + network_name + "/" + data_t + "_injection.csv"

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
