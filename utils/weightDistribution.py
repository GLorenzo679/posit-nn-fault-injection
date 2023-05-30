import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

def main():
    weights_path = PATH + "/data/CIFAR10/convnet/"

    with tf.Session() as sess:
        # This object loads the model
        load_mod = tf.train.import_meta_graph(weights_path + "posit32.ckpt.meta")

        # Loading weights and biases and other stuff to the model
        load_mod.restore(sess, weights_path + "posit32.ckpt")

        weights_l1 = sess.graph.get_tensor_by_name("Variable:0")
        weights_l2 = sess.graph.get_tensor_by_name("Variable_2:0")
        p_l1_weights_array = weights_l1.eval().ravel()
        p_l2_weights_array = weights_l2.eval().ravel()
        p_weights_array = np.append(p_l1_weights_array, p_l2_weights_array)
    
    tf.reset_default_graph()

    with tf.Session() as sess:
        # This object loads the model
        load_mod = tf.train.import_meta_graph(weights_path + "float32.ckpt.meta")

        # Loading weights and biases and other stuff to the model
        load_mod.restore(sess, weights_path + "float32.ckpt")

        weights_l1 = sess.graph.get_tensor_by_name("Variable:0")
        weights_l2 = sess.graph.get_tensor_by_name("Variable_2:0")
        f_l1_weights_array = weights_l1.eval().ravel()
        f_l2_weights_array = weights_l2.eval().ravel()
        f_weights_array = np.append(f_l1_weights_array, f_l2_weights_array)

    fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True)

    axs[0].hist(p_weights_array, bins=100, alpha=0.5, label='posit32', edgecolor="k")
    axs[0].hist(f_weights_array, bins=100, alpha=0.5, label='float32', edgecolor="k")
    axs[0].legend(loc='upper right')
    axs[1].hist(p_l1_weights_array, bins=100, alpha=0.5, label='posit32_l1', edgecolor="k")
    axs[1].hist(f_l1_weights_array, bins=100, alpha=0.5, label='float32_l1', edgecolor="k")
    axs[1].legend(loc='upper right')
    axs[2].hist(p_l2_weights_array, bins=100, alpha=0.5, label='posit32_l2', edgecolor="k")
    axs[2].hist(f_l2_weights_array, bins=100, alpha=0.5, label='float32_l2', edgecolor="k")
    axs[2].legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    main()
