import numpy as np
import tensorflow as tf
import spn.structure.Base as base
import spn.structure.leaves.parametric.Parametric as para

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import tensorflow.contrib.distributions as dists

import time


def add_to_map(given_map, key, item):
    existing_items = given_map.get(key, [])
    given_map[key] = existing_items + [item]


def variable_with_weight_decay(name, shape, stddev, wd, mean=0.0, values=None):
    if values is None:
        initializer = tf.truncated_normal_initializer(mean=mean, stddev=stddev, dtype=tf.float32)
    else:
        initializer = tf.constant_initializer(values)
    """Get a TF variable with optional l2-loss attached."""
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="weight_loss")
        tf.add_to_collection("losses", weight_decay)
        tf.add_to_collection("weight_losses", weight_decay)

    return var


def print_if_nan(tensor, msg):
    is_nan = tf.reduce_any(tf.is_nan(tensor))
    return tf.cond(is_nan, lambda: tf.Print(tensor, [is_nan], message=msg), lambda: tf.identity(tensor))


class NodeVector(object):
    def __init__(self, name):
        self.name = name

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


class SpnArgs(object):
    def __init__(self):
        self.gauss_min_var = 0.1
        self.gauss_max_var = 1.0
        self.num_gauss = 20
        self.gauss_param_l2 = None
        self.gauss_isotropic = False

        self.linear_sum_weights = False
        self.normalized_sums = True
        self.sum_weight_l2 = None
        self.num_sums = 20

        self.drop_connect = False


class GaussVector(NodeVector):
    def __init__(self, region, args, name, given_means=None, given_stddevs=None, mean=0.0):
        super().__init__(name)
        self.local_size = len(region)
        self.args = args
        self.scope = sorted(list(region))
        self.size = args.num_gauss

        self.means = variable_with_weight_decay(
            name + "_means",
            shape=[1, self.local_size, args.num_gauss],
            stddev=1e-1,
            mean=mean,
            wd=args.gauss_param_l2,
            values=given_means,
        )

        if args.gauss_min_var < args.gauss_max_var:
            if args.gauss_isotropic:
                sigma_params = variable_with_weight_decay(
                    name + "_sigma_params",
                    shape=[1, 1, args.num_gauss],
                    stddev=1e-1,
                    wd=args.gauss_param_l2,
                    values=given_stddevs,
                )
            else:
                sigma_params = variable_with_weight_decay(
                    name + "_sigma_params",
                    shape=[1, self.local_size, args.num_gauss],
                    stddev=1e-1,
                    wd=args.gauss_param_l2,
                    values=given_stddevs,
                )

            self.sigma = args.gauss_min_var + (args.gauss_max_var - args.gauss_min_var) * tf.sigmoid(sigma_params)
        else:
            self.sigma = 1.0

        self.dist = dists.Normal(self.means, tf.sqrt(self.sigma))

    def forward(self, inputs, marginalized=None):
        local_inputs = tf.gather(inputs, self.scope, axis=1)
        # gauss_log_pdf_single = - 0.5 * (tf.expand_dims(local_inputs, -1) - self.means) ** 2 / self.sigma \
        #                        - tf.log(tf.sqrt(2 * np.pi * self.sigma))
        gauss_log_pdf_single = self.dist.log_prob(tf.expand_dims(local_inputs, axis=-1))

        if marginalized is not None:
            marginalized = tf.clip_by_value(marginalized, 0.0, 1.0)
            local_marginalized = tf.expand_dims(tf.gather(marginalized, self.scope, axis=1), axis=-1)
            # weighted_gauss_pdf = (1 - local_marginalized) * tf.exp(gauss_log_pdf_single)
            # local_marginalized_broadcast = local_marginalized * tf.ones_like(weighted_gauss_pdf)

            # stacked = tf.stack([weighted_gauss_pdf, local_marginalized_broadcast], axis=3)
            # gauss_log_pdf_single = tf.log(weighted_gauss_pdf + local_marginalized_broadcast)
            gauss_log_pdf_single = gauss_log_pdf_single * (1 - local_marginalized)

        gauss_log_pdf = tf.reduce_sum(gauss_log_pdf_single, 1)
        return gauss_log_pdf

    def sample(self, num_samples, num_dims, seed=None):
        # sample_values = self.dist.sample([num_samples], seed=seed)[:, 0]
        sample_values = self.means + tf.zeros([num_samples, self.local_size, self.args.num_gauss])
        sample_shape = [num_samples, num_dims, self.size]
        indices = tf.meshgrid(tf.range(num_samples), self.scope, tf.range(self.size))
        indices = tf.stack(indices, axis=-1)
        indices = tf.transpose(indices, [1, 0, 2, 3])
        samples = tf.scatter_nd(indices, sample_values, sample_shape)
        return samples

    def num_params(self):
        result = self.means.shape.num_elements()
        if isinstance(self.sigma, tf.Tensor):
            result += self.sigma.shape.num_elements()
        return result


class ProductVector(NodeVector):
    def __init__(self, vector1, vector2, name):
        """Initialize a product vector, which takes the cross-product of two distribution vectors."""
        super().__init__(name)
        self.vector1 = vector1
        self.vector2 = vector2
        self.inputs = [vector1, vector2]

        self.scope = list(set(vector1.scope) | set(vector2.scope))

        assert len(set(vector1.scope) & set(vector2.scope)) == 0

        self.size = vector1.size * vector2.size

    def forward(self, inputs):
        dists1 = inputs[0]
        dists2 = inputs[1]
        with tf.variable_scope("products") as scope:
            num_dist1 = int(dists1.shape[1])
            num_dist2 = int(dists2.shape[1])

            # we take outer products, thus expand in different dims
            dists1_expand = tf.expand_dims(dists1, 1)
            dists2_expand = tf.expand_dims(dists2, 2)

            # product == sum in log-domain
            prod = dists1_expand + dists2_expand
            # flatten out the outer product
            prod = tf.reshape(prod, [dists1.shape[0], num_dist1 * num_dist2])

        return prod

    def num_params(self):
        return 0

    def sample(self, inputs, seed=None):
        in1_expand = tf.expand_dims(inputs[0], -1)
        in2_expand = tf.expand_dims(inputs[1], -2)

        output_shape = [inputs[0].shape[0], inputs[0].shape[1], (inputs[0].shape[2] * inputs[1].shape[2])]

        result = tf.reshape(in1_expand + in2_expand, output_shape)
        return result


class SumVector(NodeVector):
    def __init__(self, prod_vectors, num_sums, args, dropout_op=None, name="", given_weights=None):
        super().__init__(name)
        self.inputs = prod_vectors
        self.size = num_sums

        self.scope = self.inputs[0].scope

        for inp in self.inputs:
            assert set(inp.scope) == set(self.scope)

        self.dropout_op = dropout_op
        self.args = args
        num_inputs = sum([v.size for v in prod_vectors])
        self.params = variable_with_weight_decay(
            name + "_weights", shape=[1, num_inputs, num_sums], stddev=5e-1, wd=None, values=given_weights
        )
        if args.linear_sum_weights:
            if args.normalized_sums:
                self.weights = tf.nn.softmax(self.params, 1)
            else:
                self.weights = self.params ** 2
        else:
            if args.normalized_sums:
                self.weights = tf.nn.log_softmax(self.params, 1)
                if args.sum_weight_l2:
                    exp_weights = tf.exp(self.weights)
                    weight_decay = tf.multiply(tf.nn.l2_loss(exp_weights), args.sum_weight_l2)
                    tf.add_to_collection("losses", weight_decay)
                    tf.add_to_collection("weight_losses", weight_decay)
            else:
                self.weights = self.params

    def forward(self, inputs):
        prods = tf.concat(inputs, 1)
        weights = self.weights

        if self.args.linear_sum_weights:
            sums = tf.log(tf.matmul(tf.exp(prods), tf.squeeze(self.weights)))
        else:
            prods = tf.expand_dims(prods, axis=-1)
            if self.dropout_op is not None:
                if self.args.drop_connect:
                    batch_size = prods.shape[0]
                    prod_num = prods.shape[1]
                    dropout_shape = [batch_size, prod_num, self.size]

                    random_tensor = random_ops.random_uniform(dropout_shape, dtype=self.weights.dtype)
                    dropout_mask = tf.log(math_ops.floor(self.dropout_op + random_tensor))
                    weights = weights + dropout_mask

                else:
                    random_tensor = random_ops.random_uniform(prods.shape, dtype=prods.dtype)
                    dropout_mask = tf.log(math_ops.floor(self.dropout_op + random_tensor))
                    prods = prods + dropout_mask

            sums = tf.reduce_logsumexp(prods + weights, axis=1)

        return sums

    def sample(self, inputs, seed=None):
        inputs = tf.concat(inputs, 2)
        logits = tf.transpose(self.weights[0])
        dist = dists.Categorical(logits=logits)

        indices = dist.sample([inputs.shape[0]], seed=seed)
        indices = tf.reshape(tf.tile(indices, [1, inputs.shape[1]]), [inputs.shape[0], self.size, inputs.shape[1]])
        indices = tf.transpose(indices, [0, 2, 1])

        others = tf.meshgrid(tf.range(inputs.shape[1]), tf.range(inputs.shape[0]), tf.range(self.size))

        indices = tf.stack([others[1], others[0], indices], axis=-1)

        result = tf.gather_nd(inputs, indices)
        return result

    def num_params(self):
        return self.weights.shape.num_elements()


class RatSpn(object):
    def __init__(self, num_classes, region_graph=None, vector_list=None, args=SpnArgs(), name=None, mean=0.0):
        if name is None:
            name = str(id(self))
        self.name = name
        self._region_graph = region_graph
        self.args = args
        self.default_mean = mean

        self.num_classes = num_classes
        # self.num_dims = len(self._region_graph.get_root_region())

        # dictionary mapping regions to tensor of sums/input distributions
        self._region_distributions = dict()
        # dictionary mapping regions to tensor of products
        self._region_products = dict()

        self.vector_list = []
        self.output_vector = None

        # make the SPN...
        with tf.variable_scope(self.name) as scope:
            if region_graph is not None:
                self._make_spn_from_region_graph()
            elif vector_list is not None:
                self._make_spn_from_vector_list(vector_list)
            else:
                raise ValueError("Either vector_list or region_graph must not be None")

        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.num_dims = len(self.output_vector.scope)

    def _make_spn_from_vector_list(self, vector_list):
        self.vector_list = [[]]

        node_to_vec = {}

        for i, leaf_vector in enumerate(vector_list[0]):
            name = "gauss_{}".format(i)
            scope = leaf_vector[0].scope
            num_gauss = len(leaf_vector)
            means = np.zeros((len(scope), num_gauss))
            stdevs = np.zeros((len(scope), num_gauss))
            for j, prod_node in enumerate(leaf_vector):
                for k, gauss_node in enumerate(prod_node.children):
                    means[k, j] = gauss_node.mean
                    stdevs[k, j] = gauss_node.stdev

            gauss_vector = GaussVector(scope, self.args, name, given_means=means, given_stddevs=stdevs)
            self.vector_list[0].append(gauss_vector)

            for j, prod_node in enumerate(leaf_vector):
                node_to_vec[id(prod_node)] = gauss_vector

        for layer_num, layer in enumerate(vector_list[1:]):
            self.vector_list.append([])
            for vector_num, vector in enumerate(layer):
                if type(vector[0]) == base.Product:
                    child_vec1 = node_to_vec[id(vector[0].children[0])]
                    child_vec2 = node_to_vec[id(vector[0].children[1])]
                    name = "prod_{}_{}".format(layer_num, vector_num)
                    new_vector = ProductVector(child_vec1, child_vec2, name)
                elif type(vector[0]) == base.Sum:
                    child_vecs = list(set([node_to_vec[id(child_node)] for child_node in vector[0].children]))
                    assert len(child_vecs) <= 2
                    name = "sum_{}_{}".format(layer_num, vector_num)
                    num_inputs = sum([v.size for v in child_vecs])
                    weights = np.zeros((num_inputs, len(vector)))
                    for node_num, node in enumerate(vector):
                        weights[:, node_num] = node.weights
                    new_vector = SumVector(child_vecs, len(vector), self.args, name=name, given_weights=weights)
                else:
                    assert False

                self.vector_list[-1].append(new_vector)

                for node in vector:
                    node_to_vec[id(node)] = new_vector

        self.output_vector = self.vector_list[-1][-1]

    def _make_spn_from_region_graph(self):
        """Build a RAT-SPN."""

        rg_layers = self._region_graph.make_layers()
        self.rg_layers = rg_layers

        # make leaf layer (always Gauss currently)
        self.vector_list.append([])
        for i, leaf_region in enumerate(rg_layers[0]):
            name = "gauss_{}".format(i)
            gauss_vector = GaussVector(leaf_region, self.args, name, mean=self.default_mean)
            self.vector_list[-1].append(gauss_vector)
            self._region_distributions[leaf_region] = gauss_vector

        # make sum-product layers
        ps_count = 0
        for layer_idx in range(1, len(rg_layers)):
            self.vector_list.append([])
            if layer_idx % 2 == 1:
                partitions = rg_layers[layer_idx]
                for i, partition in enumerate(partitions):
                    input_regions = list(partition)
                    input1 = self._region_distributions[input_regions[0]]
                    input2 = self._region_distributions[input_regions[1]]
                    vector_name = "prod_{}_{}".format(layer_idx, i)
                    prod_vector = ProductVector(input1, input2, vector_name)
                    self.vector_list[-1].append(prod_vector)

                    resulting_region = frozenset(input_regions[0] | input_regions[1])
                    add_to_map(self._region_products, resulting_region, prod_vector)
            else:
                cur_num_sums = self.num_classes if layer_idx == len(rg_layers) - 1 else self.args.num_sums

                regions = rg_layers[layer_idx]
                for i, region in enumerate(regions):
                    product_vectors = self._region_products[region]
                    vector_name = "sum_{}_{}".format(layer_idx, i)
                    sum_vector = SumVector(product_vectors, cur_num_sums, self.args, name=vector_name)
                    self.vector_list[-1].append(sum_vector)

                    self._region_distributions[region] = sum_vector

                ps_count = ps_count + 1

        self.output_vector = self._region_distributions[self._region_graph.get_root_region()]

    def forward(self, inputs, marginalized=None):
        obj_to_tensor = {}
        for leaf_vector in self.vector_list[0]:
            obj_to_tensor[leaf_vector] = leaf_vector.forward(inputs, marginalized)

        for layer_idx in range(1, len(self.vector_list)):
            for vector in self.vector_list[layer_idx]:
                input_tensors = [obj_to_tensor[obj] for obj in vector.inputs]
                result = vector.forward(input_tensors)
                obj_to_tensor[vector] = result

        return obj_to_tensor[self.output_vector]

    # Does not work currently!
    def sample(self, num_samples=10, seed=None):
        vec_to_samples = {}
        for leaf_vector in self.vector_list[0]:
            vec_to_samples[leaf_vector] = leaf_vector.sample(num_samples, self.num_dims, seed=seed)

        for layer_idx in range(1, len(self.vector_list)):
            for vector in self.vector_list[layer_idx]:
                input_samples = [vec_to_samples[vec] for vec in vector.inputs]
                result = vector.sample(input_samples, seed=seed)
                vec_to_samples[vector] = result

        return vec_to_samples[self.output_vector]

    def num_params(self):
        result = 0
        for layer in self.vector_list:
            for vector in layer:
                result += vector.num_params()

        return result

    def get_simple_spn(self, sess, single_root=False):
        start_time = time.time()
        vec_to_params = {}
        for leaf_vector in self.vector_list[0]:
            vec_to_params[leaf_vector] = (leaf_vector.means[0], leaf_vector.sigma[0])
        for layer_idx in range(1, len(self.vector_list)):
            if layer_idx % 2 == 0:
                for sum_vec in self.vector_list[layer_idx]:
                    vec_to_params[sum_vec] = sum_vec.weights[0]

        st = time.time()
        vec_to_params = sess.run(vec_to_params)
        time_tf = time.time() - st

        vec_to_nodes = {}
        node_id = -1

        for leaf_vector in self.vector_list[0]:
            vec_to_nodes[leaf_vector] = []
            means, sigmas = vec_to_params[leaf_vector]
            stdevs = np.sqrt(sigmas) + np.zeros_like(means)  # Use broadcasting to expand stdev is necessary
            for i in range(leaf_vector.size):
                prod = base.Product()
                prod.id = node_id = node_id + 1
                prod.scope.extend(leaf_vector.scope)
                for j, r in enumerate(leaf_vector.scope):
                    gaussian = para.Gaussian(mean=means[j, i], stdev=stdevs[j, i], scope=[r])
                    gaussian.id = node_id = node_id + 1
                    prod.children.append(gaussian)

                vec_to_nodes[leaf_vector].append(prod)

        for layer_idx in range(1, len(self.vector_list)):
            # vector_list.append([])
            if layer_idx % 2 == 1:
                prod_vectors = self.vector_list[layer_idx]
                for i, prod_vector in enumerate(prod_vectors):
                    input1 = prod_vector.vector1
                    input2 = prod_vector.vector2

                    vec_to_nodes[prod_vector] = []

                    # The order of these loops is very important, otherwise weights will be mismatched
                    # input1 is the inner loop because it is the inner dimension
                    # of the outer product in ProductVector::forward
                    for c2 in range(input2.size):
                        for c1 in range(input1.size):
                            prod = base.Product()
                            prod.id = node_id = node_id + 1
                            prod.children.append(vec_to_nodes[input1][c1])
                            prod.children.append(vec_to_nodes[input2][c2])
                            prod.scope.extend(input1.scope)
                            prod.scope.extend(input2.scope)
                            vec_to_nodes[prod_vector].append(prod)

            else:
                sum_vectors = self.vector_list[layer_idx]
                for i, sum_vector in enumerate(sum_vectors):
                    vec_to_nodes[sum_vector] = []
                    weights = vec_to_params[sum_vector]

                    for j in range(sum_vector.size):
                        sum_node = base.Sum()
                        if layer_idx < len(self.vector_list) - 1:
                            sum_node.id = node_id = node_id + 1
                        else:
                            sum_node.id = node_id + 1
                        sum_node.scope.extend(sum_vector.scope)
                        input_vecs = [vec_to_nodes[prod_vec] for prod_vec in sum_vector.inputs]
                        input_nodes = [node for vec in input_vecs for node in vec]
                        sum_node.children.extend(input_nodes)

                        vec_to_nodes[sum_vector].append(sum_node)

                        log_weights = weights[:, j]
                        scaled_weights = np.exp(log_weights - np.max(log_weights))
                        normalized_weights = scaled_weights / np.sum(scaled_weights)
                        sum_node.weights.extend(normalized_weights)

        output_nodes = vec_to_nodes[self.output_vector]

        if single_root:
            for i, node in enumerate(output_nodes):
                node.id = node.id + i
                node_id += 1
            root = base.Sum()
            root.id = node_id = node_id + 1
            root.children.extend(output_nodes)
            root.scope.extend(output_nodes[0].scope)
            root.weights.extend([1.0 / float(len(output_nodes))] * len(output_nodes))
            return root

        print("conversion finished in {:3f}s".format(time.time() - start_time))
        print("time spent evaluating by Tensorflow: {:3f}s".format(time_tf))
        return output_nodes


def compute_performance(sess, data_x, data_labels, batch_size, spn):
    """Compute classification accuracy"""

    num_batches = int(np.ceil(float(data_x.shape[0]) / float(batch_size)))
    test_idx = 0
    num_correct = 0

    for test_k in range(0, num_batches):
        if test_k + 1 < num_batches:
            batch_data = data_x[test_idx : test_idx + batch_size, :]
            batch_labels = data_labels[test_idx : test_idx + batch_size]

        feed_dict = {spn.inputs: batch_data, spn.labels: batch_labels}
        if spn.dropout_input_placeholder is not None:
            feed_dict[spn.dropout_input_placeholder] = 1.0
        for dropout_op in spn.dropout_layer_placeholders:
            if dropout_op is not None:
                feed_dict[dropout_op] = 1.0

        spn_outputs = sess.run(spn.outputs, feed_dict=feed_dict)
        max_output = np.argmax(spn_outputs, axis=1)

        num_correct_batch = np.sum(max_output == batch_labels)

        num_correct += num_correct_batch

        test_idx += batch_size

    accuracy = num_correct / (num_batches * batch_size)

    return accuracy
