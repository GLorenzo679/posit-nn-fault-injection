from __future__ import division
from __future__ import print_function

import numpy as np
import softposit as sp
import tensorflow as tf
import sys
import os
import random


def random_mask_generator(size):
    bits = np.random.randint(2, size=size)
    mask = "0b" + str(bits)[1:-1].replace(" ", "")

    return mask

def injection(np_posit: int, bit_size: int, pos: int):
    """
    Takes as input a posit number and flips one of its bit in a given position.

    Args:
        np_posit:   A numpy-posit number
        bit_size:   Bit size of posit input number
        pos:        Bit position of bit flip
    Return:
        inj_posit:  Injected posit number
    """

    try:
        if bit_size not in (8, 16, 32):
            raise Exception("\nBit size should be one between: (8, 16, 32)")
    except Exception as e:
        print('Caught Exception:', e)
        sys.exit()

    if bit_size == 8:
        posit = sp.posit8()
        posit_mask = sp.posit8()
    elif bit_size == 16:
        posit = sp.posit16()
        posit_mask = sp.posit16()
    else:
        posit = sp.posit32()
        posit_mask = sp.posit32()


def main():
    injection(12, int(sys.argv[1]))

    with tf.Session() as sess:
        path = os.getcwd() + "/deep-pensieve/src/TensorFlow/data/CIFAR10/"

        # This object loads the model
        LoadMod = tf.train.import_meta_graph(path + "posit8.ckpt.meta")

        # Loading weights and biases and other stuff to the model
        path2 = LoadMod.restore(sess, tf.train.latest_checkpoint(path))

        # print(tf.train.list_variables(path))

        w1 = sess.graph.get_tensor_by_name("Variable:0")
        # print('normal tensor: \n')
        # print(sess.run(w1))

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

        # binary representation of posit8 number before injection:
        print("Binary representation of posit8 number before injection:")
        print(
            bin(int.from_bytes(fault[rand_index].tobytes(), byteorder=sys.byteorder)) + "\n"
        )

        # create softposit posit8 object
        sp8 = sp.posit8()
        # extract binary representation of np posit8
        np8_bin_representation = bin(
            int.from_bytes(fault[rand_index].tobytes(), byteorder=sys.byteorder)
        )
        # create a softposit posit8 with bites from np posit8
        sp8.fromBits(int(np8_bin_representation, 2))

        # fault injection
        sp8_mask = sp.posit8()
        mask = random_mask_generator(size=8)
        print(f"Mask: {mask}")
        sp8_mask.fromBits(int(mask, 0))
        # sp8_mask = random_mask_generator(8)
        sp8 = sp8 ^ sp8_mask

        # assign to previous np posit8 new sp posit8
        fault[rand_index] = sp8

        print(f"New posit: {fault[rand_index]}\n")

        # binary representation of posit8 number after injection:
        print("Binary representation of posit8 number after injection:")
        print(bin(int.from_bytes(fault[rand_index].tobytes(), byteorder=sys.byteorder)))

        # print('\nmodified tensor: \n')
        sess.run(tf.assign(w1, fault))
        # print(sess.run(w1))

main()