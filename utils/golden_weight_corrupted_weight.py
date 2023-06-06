import os
import csv

from src.Injection import Injection
from utils.utils import (
    get_evaluator,
    get_inference,
    get_loader,
    get_network,
    get_network_parameters,
    get_sp_type,
    get_name_file,
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
        num_layer  ,  # num_layer limited to convolutional layers only for now
        tensor_shape,
        num_bit_representation=args.bit_len,
        type=get_sp_type(data_t),
        number_of_faults=args.force_n,
        net_level = args.net_level,
        bit_index_low = args.low_index,
        bit_index_high = args.high_index
    )
    
    # setup path for results file
    PATH = os.path.abspath(os.path.dirname(__file__))
    results_path = PATH + "/res/" + data_set + "/" + network_name + "/" +  get_name_file(data_t, args.name_output)

    set_weights(injection)
    # perform inference for every fault in fault list
    for fault in injection.fault_list:
        with open(results_path, "a+") as file:
            headers = [
                "fault_id",
                "layer_index",
                "tensor_index",
                "bit_index",
                "golden_weight"
                "corrupted_weight"
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
                "golden_weight": fault.weight_start,
                "corrputed_weight": fault.weight_corrupted,
                "weight_difference": fault.weight_start - fault.weight_corrupted,
            }
        )

# def set_weights(injection):
#     for fault in injection.fault_list:
#         np_weights = weights.eval()
#         fault.set_weight(np_weights[fault.tensor_index])


if __name__ == "__main__":
    main(args=parse_args())
