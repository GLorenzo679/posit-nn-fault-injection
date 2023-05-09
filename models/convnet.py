import tensorflow as tf
from tensorflow.contrib.layers import flatten


def model(x, tf_type):
    """## Implementation of CifarNet
    Implements the [CifarNet](https://github.com/tensorflow/models/blob/master/research/slim/nets/cifarnet.py)

    # Input
    The CifarNet architecture accepts a 32x32xC image as input, where C is the number of color channels. Since CIFAR10 images are RGB, C is 3 in this case.

    # Output
    Return the forwarded prediction - logits.
    """

    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x64.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 64), mean=mu, stddev=sigma, dtype=tf_type))
    conv1_b = tf.Variable(tf.zeros(64, dtype=tf_type))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding="SAME") + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 32x32x64. Output = 16x16x64.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Local Response Normalization
    # Use BatchNorm instead

    # Batch Normalization
    # conv1 = batch_norm(conv1, 64, training)
    # conv1 = tf.compat.v1.layers.BatchNormalization()(conv1)

    # Layer 2: Convolutional. Output = 16x16x64.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 64, 64), mean=mu, stddev=sigma, dtype=tf_type))
    conv2_b = tf.Variable(tf.zeros(64, dtype=tf_type))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding="SAME") + conv2_b

    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Local Response Normalization
    # conv2 = tf.cast(conv2, tf.float32)
    # norm2 = tf.nn.lrn(conv2, 4, bias=posit(1.0), alpha=posit(0.001 / 9.0), beta=posit(0.75))
    # norm2 = tf.cast(norm2, tf_type)

    # Batch Normalization
    # conv2 = batch_norm(conv2, 64, training)

    # Pooling. Input = 16x16x64. Output = 8x8x64.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

    # Dropout
    # conv2 = tf.nn.dropout(conv2, keep_prob)

    # Flatten. Input = 8x8x64. Output = 4096.
    fc0 = flatten(conv2)
    dim = fc0.get_shape()[1].value

    # Layer 3: Fully Connected. Input = 4096. Output = 384.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(dim, 384), mean=mu, stddev=sigma, dtype=tf_type))
    fc1_b = tf.Variable(tf.zeros(384, dtype=tf_type))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # Activation.
    fc1 = tf.nn.relu(fc1)

    # Dropout
    # fc1 = tf.nn.dropout(fc1, keep_prob)

    # Layer 4: Fully Connected. Input = 384. Output = 192.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(384, 192), mean=mu, stddev=sigma, dtype=tf_type))
    fc2_b = tf.Variable(tf.zeros(192, dtype=tf_type))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # Activation.
    fc2 = tf.nn.relu(fc2)

    # Dropout
    # fc2 = tf.nn.dropout(fc2, keep_prob)

    # Layer 5: Linear layer(WX + b). Input = 192. Output = 10.
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(192, 10), mean=mu, stddev=sigma, dtype=tf_type))
    fc3_b = tf.Variable(tf.zeros(10, dtype=tf_type))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits


def evaluate(X_data, y_data, batch_size, x, y, training, accuracy_operation, top5_operation):
    num_examples = len(X_data)
    total_accuracy = 0
    total_top5 = 0
    sess = tf.get_default_session()

    for offset in range(0, num_examples, batch_size):
        print(f"\nBatch {int(offset/batch_size + 1)}:")
        batch_x, batch_y = (
            X_data[offset : offset + batch_size],
            y_data[offset : offset + batch_size],
        )
        accuracy, top5 = sess.run(
            [accuracy_operation, top5_operation],
            feed_dict={x: batch_x, y: batch_y, training: False},
        )

        print(f"\tAccuracy: {accuracy}\tTop-5: {top5}")

        total_accuracy += accuracy * len(batch_x)
        total_top5 += top5 * len(batch_x)
    return (total_accuracy / num_examples, total_top5 / num_examples)
