from observations import mnist
import tensorflow as tf
import numpy as np
import spn.experiments.RandomSPNs.RAT_SPN as RAT_SPN
import spn.experiments.RandomSPNs.region_graph as region_graph

import spn.algorithms.Inference as inference
import spn.io.Graphics as graphics


def one_hot(vector):
    result = np.zeros((vector.size, vector.max() + 1))
    result[np.arange(vector.size), vector] = 1
    return result


def load_mnist():
    (train_im, train_lab), (test_im, test_lab) = mnist("data/mnist")
    train_im_mean = np.mean(train_im, 0)
    train_im_std = np.std(train_im, 0)
    std_eps = 1e-7
    train_im = (train_im - train_im_mean) / (train_im_std + std_eps)
    test_im = (test_im - train_im_mean) / (train_im_std + std_eps)

    # train_im /= 255.0
    # test_im /= 255.0
    return (train_im, train_lab), (test_im, test_lab)


def train_spn(spn, train_im, train_lab=None, num_epochs=50, batch_size=100, sess=tf.Session()):

    input_ph = tf.placeholder(tf.float32, [batch_size, train_im.shape[1]])
    label_ph = tf.placeholder(tf.int32, [batch_size])
    marginalized = tf.zeros_like(input_ph)
    spn_output = spn.forward(input_ph, marginalized)
    if train_lab is not None:
        disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_ph, logits=spn_output))
        label_idx = tf.stack([tf.range(batch_size), label_ph], axis=1)
        gen_loss = tf.reduce_mean(-1 * tf.gather_nd(spn_output, label_idx))
    very_gen_loss = -1 * tf.reduce_mean(tf.reduce_logsumexp(spn_output, axis=1))
    loss = disc_loss
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    batches_per_epoch = train_im.shape[0] // batch_size

    # sess.run(tf.variables_initializer(optimizer.variables()))
    sess.run(tf.global_variables_initializer())

    for i in range(num_epochs):
        num_correct = 0
        for j in range(batches_per_epoch):
            im_batch = train_im[j * batch_size : (j + 1) * batch_size, :]
            label_batch = train_lab[j * batch_size : (j + 1) * batch_size]

            _, cur_output, cur_loss = sess.run(
                [train_op, spn_output, loss], feed_dict={input_ph: im_batch, label_ph: label_batch}
            )

            max_idx = np.argmax(cur_output, axis=1)

            num_correct_batch = np.sum(max_idx == label_batch)
            num_correct += num_correct_batch

        acc = num_correct / (batch_size * batches_per_epoch)
        print(i, acc, cur_loss)


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


if __name__ == "__main__":
    rg = region_graph.RegionGraph(range(28 * 28))
    # rg = region_graph.RegionGraph(range(3 * 3))
    for _ in range(0, 2):
        rg.random_split(2, 1)

    args = RAT_SPN.SpnArgs()
    args.normalized_sums = True
    args.num_sums = 2
    args.num_gauss = 2
    spn = RAT_SPN.RatSpn(10, region_graph=rg, name="obj-spn", args=args)
    print("num_params", spn.num_params())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    (train_im, train_labels), _ = load_mnist()
    train_spn(spn, train_im, train_labels, num_epochs=3, sess=sess)

    # dummy_input = np.random.normal(0.0, 1.2, [10, 9])
    dummy_input = train_im[:5]
    input_ph = tf.placeholder(tf.float32, dummy_input.shape)
    output_tensor = spn.forward(input_ph)
    tf_output = sess.run(output_tensor, feed_dict={input_ph: dummy_input})

    output_nodes = spn.get_simple_spn(sess)
    simple_output = []
    for node in output_nodes:
        simple_output.append(inference.log_likelihood(node, dummy_input)[:, 0])
    # graphics.plot_spn2(output_nodes[0])
    # graphics.plot_spn_to_svg(output_nodes[0])
    simple_output = np.stack(simple_output, axis=-1)
    print(tf_output, simple_output)
    simple_output = softmax(simple_output, axis=1)
    tf_output = softmax(tf_output, axis=1) + 1e-100
    print(tf_output, simple_output)
    relative_error = np.abs(simple_output / tf_output - 1)
    print(np.average(relative_error))
