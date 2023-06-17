from __future__ import division, print_function

import os
import random
import sys

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

        w1 = sess.graph.get_tensor_by_name("Variable:0")

        # create ndarray of tensor
        fault = w1.eval()
        # select random index in ndarray
        rand_index = (
            random.randrange(0, 5),
            random.randrange(0, 5),
            random.randrange(0, 3),
            random.randrange(0, 64),
        )

        print(f"Number present at random index: {rand_index}")
        print(fault[rand_index])

        # binary representation of posit32 number before injection:
        print("Binary representation of posit32 number before injection:")
        print(bin(int.from_bytes(fault[rand_index].tobytes(), byteorder=sys.byteorder)) + "\n")

        # create softposit posit32 object
        sp32 = sp.posit32()
        # extract binary representation of np posit32
        np32_bin_representation = bin(int.from_bytes(fault[rand_index].tobytes(), byteorder=sys.byteorder))
        # create a softposit posit32 with bites from np posit32
        sp32.fromBits(int(np32_bin_representation, 2))

        # fault injection
        sp32_mask = sp.posit32()
        mask = random_bit_mask_generator(32, 31)
        print(f"Mask:\n{mask}")
        sp32_mask.fromBits(int(mask, 0))
        sp32 = sp32 ^ sp32_mask

        # assign to previous np posit32 new sp posit32
        fault[rand_index] = sp32

        print(f"\nNew posit: {fault[rand_index]}")

        # binary representation of posit32 number after injection:
        print("Binary representation of posit32 number after injection:")
        print(bin(int.from_bytes(fault[rand_index].tobytes(), byteorder=sys.byteorder)))


if __name__ == "__main__":
    main()
