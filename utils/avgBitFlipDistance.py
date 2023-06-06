from __future__ import division, print_function

import os
import sys
import csv

from ast import literal_eval as make_tuple
import softposit as sp
import tensorflow as tf

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


def random_bit_mask_generator(num_bit_representation, bit_index):
    return "0b" + "0" * (num_bit_representation - 1 - bit_index) + str(1) + "0" * (bit_index)


def main():
    weights_path = PATH + "/data/CIFAR10/convnet/"

    with tf.Session() as sess:
        # This object loads the model
        load_mod = tf.train.import_meta_graph(weights_path + "posit32.ckpt.meta")

        # Loading weights and biases and other stuff to the model
        load_mod.restore(sess, weights_path + "posit32.ckpt")

        for l_net in range(2):
            for bit in range(29, 32):
                with open(PATH + "/res/CIFAR10/convnet/posit32_net_level_" + l_net + "_bit_" + bit + ".csv") as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    next(csv_reader)

                    Tot_BFD = 0

                    for row in csv_reader:
                        layer_index = int(row[1])
                        tensor_index = make_tuple(row[2])
                        bit_index = int(row[3])

                        if(layer_index == 0):
                            weight = sess.graph.get_tensor_by_name("Variable:0")
                        else:
                            weight = sess.graph.get_tensor_by_name("Variable_2:0")

                        fault = weight.eval()

                        weight_start = fault[tensor_index]

                        print(f"Number present at random index: {tensor_index}")
                        print(weight_start)

                        # create softposit posit32 object
                        weight_corrupted = sp.posit32()
                        # extract binary representation of np posit32
                        np32_bin_representation = bin(int.from_bytes(weight_start.tobytes(), byteorder=sys.byteorder))
                        # create a softposit posit32 with bites from np posit32
                        weight_corrupted.fromBits(int(np32_bin_representation, 2))

                        # fault injection
                        sp32_mask = sp.posit32()
                        mask = random_bit_mask_generator(32, bit_index)
                        print(f"Mask:\n{mask}")
                        sp32_mask.fromBits(int(mask, 0))
                        weight_corrupted = weight_corrupted ^ sp32_mask

                        print(f"\nNew posit: {weight_corrupted}")

                        BFD = abs(weight_start - weight_corrupted)
                        Tot_BFD += BFD

                        print(f"Bit Flip Distance: {BFD}")
                        print("-" * 50)

                        with open(PATH + "/res/CIFAR10/convnet/BFD/posit32_net_level" + l_net + "_bit_" + bit + ".csv", "a+") as file:
                            headers = [
                                "layer_index",
                                "tensor_index",
                                "bit_index",
                                "weight_start",
                                "weight_corrupted",
                                "weight_difference",
                                "bit_flip_distance"
                            ]

                            writer = csv.DictWriter(file, delimiter=",", lineterminator="\n", fieldnames=headers)

                            # writer.writeheader()

                            writer.writerow(
                                {
                                    "layer_index": layer_index,
                                    "tensor_index": tensor_index,
                                    "bit_index": bit_index,
                                    "weight_start": weight_start,
                                    "weight_corrupted": weight_corrupted,
                                    "weight_difference": abs(weight_start - weight_corrupted),
                                    "bit_flip_distance": BFD
                                }
                            )

                    print(f"Total Bit Flip Distance: {Tot_BFD}")
                    print(f"Average Bit Flip Distance: {Tot_BFD / 100}")


if __name__ == "__main__":
    main()
