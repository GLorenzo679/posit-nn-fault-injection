from __future__ import division, print_function

import csv
import os
import sys
import numpy as np
from ast import literal_eval as make_tuple
import struct

import softposit as sp
import tensorflow as tf

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def random_bit_mask_generator(num_bit_representation, bit_index):
    return "0b" + "0" * (num_bit_representation - 1 - bit_index) + str(1) + "0" * (bit_index)


def main():
    weights_path = PATH + "/data/CIFAR10/convnet/"

    with tf.Session() as sess:
        # This object loads the model
        load_mod = tf.train.import_meta_graph(weights_path + "float32.ckpt.meta")

        # Loading weights and biases and other stuff to the model
        load_mod.restore(sess, weights_path + "float32.ckpt")

        for layer_index in range(2):
            for bit_index in range(32):
                if layer_index == 0:
                    filename = PATH + "/fault_list_layer_0.csv"
                else:
                    filename = PATH + "/fault_list_layer_1.csv"

                with open(filename) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")

                    row_count = sum(1 for row in csv_reader) - 1
                    Tot_BFD = 0

                    csv_file.seek(0)

                    i = 0

                    for row in csv_reader:
                        tensor_index = make_tuple(row[1])

                        if layer_index == 0:
                            weight = sess.graph.get_tensor_by_name("Variable:0")
                        else:
                            weight = sess.graph.get_tensor_by_name("Variable_2:0")

                        fault = weight.eval()

                        weight_start = fault[tensor_index]

                        print(f"Number present at random index: {tensor_index}")
                        print(weight_start)

                        # binary representation of float32 number before injection:
                        print("Binary representation of float32 number before injection:")
                        print(bin(int.from_bytes(fault[tensor_index].tobytes(), byteorder=sys.byteorder)) + "\n")

                        # create softposit float32 object
                        #weight_corrupted = np.float32()
                        # extract binary representation of np float32
                        #np32_bin_representation = bin(int.from_bytes(weight_start.tobytes(), byteorder=sys.byteorder))
                        # create a softposit float32 with bites from np float32
                        #weight_corrupted.fromBits(int(np32_bin_representation, 2))

                        # fault injection
                        mask = random_bit_mask_generator(32, bit_index)
                        print(f"Mask:\n{mask}")
                        weight_corrupted = weight_corrupted = np.float32(
                            struct.unpack(
                                "f",
                                struct.pack(
                                    "I",
                                    int.from_bytes(fault[tensor_index].tobytes(), byteorder=sys.byteorder) ^ int(mask, 0),
                                ),
                            )[0]
                        )

                        print(f"\nNew posit: {weight_corrupted}")

                        fault[tensor_index] = weight_corrupted

                        # binary representation of float32 number after injection:
                        print("Binary representation of float32 number after injection:")
                        print(bin(int.from_bytes(fault[tensor_index].tobytes(), byteorder=sys.byteorder)))            

                        BFD = abs(weight_start - weight_corrupted)
                        Tot_BFD += BFD

                        print(f"Bit Flip Distance: {BFD}")
                        print("-" * 50)

                        with open(
                            PATH
                            + "/res/CIFAR10/convnet/BFD/float32_net_level_"
                            + str(layer_index)
                            + "_bit_"
                            + str(bit_index)
                            + ".csv",
                            "a+",
                        ) as file:
                            headers = [
                                "layer_index",
                                "tensor_index",
                                "bit_index",
                                "weight_start",
                                "weight_corrupted",
                                "weight_difference",
                                "bit_flip_distance",
                            ]

                            writer = csv.DictWriter(file, delimiter=",", lineterminator="\n", fieldnames=headers)

                            if i == 0:
                                writer.writeheader()

                            writer.writerow(
                                {
                                    "layer_index": layer_index,
                                    "tensor_index": tensor_index,
                                    "bit_index": bit_index,
                                    "weight_start": weight_start,
                                    "weight_corrupted": weight_corrupted,
                                    "weight_difference": abs(weight_start - weight_corrupted),
                                    "bit_flip_distance": BFD,
                                }
                            )

                            i += 1

                    print(f"Total Bit Flip Distance: {Tot_BFD}")
                    print(f"Average Bit Flip Distance: {Tot_BFD / row_count}")


if __name__ == "__main__":
    main()
