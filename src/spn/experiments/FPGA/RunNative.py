"""
Created on March 26, 2018

@author: Alejandro Molina
"""
import glob
import os
import platform
import subprocess
from collections import OrderedDict

import numpy as np
from natsort import natsorted

from spn.algorithms.Inference import likelihood
from spn.experiments.FPGA.GenerateSPNs import load_spn_from_file, fpga_count_ops
from spn.gpu.TensorFlow import spn_to_tf_graph
from spn.structure.Base import get_nodes_by_type, Node, get_number_of_edges, get_depth, Product, Leaf, Sum

np.set_printoptions(precision=50)

import time


def sum_to_tf_graph(node, children, data_placeholder, **args):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        return tf.add_n([node.weights[i] * ctf for i, ctf in enumerate(children)])


def prod_to_tf_graph(node, children, data_placeholder, **args):
    with tf.variable_scope("%s_%s" % (node.__class__.__name__, node.id)):
        prod_res = None
        for c in children:
            if prod_res is None:
                prod_res = c
            else:
                prod_res = tf.multiply(prod_res, c)
        return prod_res


_node_tf_graph = {Sum: sum_to_tf_graph, Product: prod_to_tf_graph, Histogram: histogram_to_tf_graph}


path = os.path.dirname(__file__)
OS_name = platform.system()


def run_experiment(exp, spn, test_data, test_type, exp_lambda):

    outprefix = path + "/spns/%s/" % (exp)

    results_file = "%stime_test_%s_ll_%s.txt" % (outprefix, test_type, OS_name)
    if os.path.isfile(results_file):
        return

    print(exp, test_data.shape, test_type)

    ll, test_time = exp_lambda()
    np.savetxt(results_file, ll, delimiter=";")

    import cpuinfo

    machine = cpuinfo.get_cpu_info()["brand"]

    adds, muls = fpga_count_ops(spn)

    test_n = test_data.shape[0]

    results = OrderedDict()
    results["Experiment"] = exp
    results["OS"] = OS_name
    results["machine"] = machine
    results["test type"] = test_type
    results["expected adds"] = adds
    results["expected muls"] = muls
    results["input rows"] = test_n
    results["input cols"] = test_data.shape[1]
    results["spn nodes"] = len(get_nodes_by_type(spn, Node))
    results["spn sum nodes"] = len(get_nodes_by_type(spn, Sum))
    results["spn prod nodes"] = len(get_nodes_by_type(spn, Product))
    results["spn leaves"] = len(get_nodes_by_type(spn, Leaf))
    results["spn edges"] = get_number_of_edges(spn)
    results["spn layers"] = get_depth(spn)
    results["time per task"] = test_time
    results["time per instance"] = test_time / test_n
    results["avg ll"] = np.mean(ll, dtype=np.float128)

    results_file_name = "results.csv"

    if not os.path.isfile(results_file_name):
        results_file = open(results_file_name, "w")
        results_file.write(";".join(results.keys()))
        results_file.write("\n")
    else:
        results_file = open(results_file_name, "a")

    results_file.write(";".join(map(str, results.values())))
    results_file.write("\n")
    results_file.close()


if __name__ == "__main__":

    for exp in natsorted(map(os.path.basename, glob.glob(path + "/spns/*"))):

        outprefix = path + "/spns/%s/" % (exp)

        spn, words, _ = load_spn_from_file(outprefix)

        print(exp, fpga_count_ops(spn))

        data = np.loadtxt(outprefix + "all_data.txt", delimiter=";")

        if data.shape[0] < 10000:
            r = np.random.RandomState(17)
            test_data = data[r.choice(data.shape[0], 10000), :]
        else:
            test_data = data

        test_data_fname = outprefix + "time_test_data.txt"

        if not os.path.isfile(test_data_fname):
            np.savetxt(test_data_fname, test_data, delimiter=";", header=";".join(words))

        def execute_tf():
            import tensorflow as tf
            from tensorflow.python.client import timeline
            import json

            tf.reset_default_graph()

            elapsed = 0
            data_placeholder = tf.placeholder(tf.int32, test_data.shape)
            tf_graph = spn_to_tf_graph(spn, data_placeholder, log_space=False)
            tfstart = time.perf_counter()
            n_repeats = 1000
            with tf.Session() as sess:

                for i in range(n_repeats):

                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()

                    sess.run(tf.global_variables_initializer())
                    # start = time.perf_counter()
                    tf_ll = sess.run(
                        tf_graph,
                        feed_dict={data_placeholder: test_data},
                        options=run_options,
                        run_metadata=run_metadata,
                    )

                    continue
                    # end = time.perf_counter()

                    # e2 = end - start

                    ctf = timeline.Timeline(run_metadata.step_stats).generate_chrome_trace_format()

                    rfile_path = outprefix + "tf_timelines2/time_line_%s.json" % i
                    if not os.path.exists(os.path.dirname(rfile_path)):
                        os.mkdir(os.path.dirname(rfile_path))
                    results_file = open(rfile_path, "w")
                    results_file.write(ctf)
                    results_file.close()

                    traceEvents = json.loads(ctf)["traceEvents"]
                    run_time = max([o["ts"] + o["dur"] for o in traceEvents if "ts" in o and "dur" in o]) - min(
                        [o["ts"] for o in traceEvents if "ts" in o]
                    )
                    run_time *= 1000

                    if i > 0:
                        # the first run is 10 times slower for whatever reason
                        elapsed += run_time

                    # if i % 20 == 0:
                    # print(exp, i, e2, run_time)
            tfend = time.perf_counter()
            tfelapsed = (tfend - tfstart) * 1000000000

            return np.log(tf_ll), tfelapsed / (n_repeats - 1)

        run_experiment(exp, spn, test_data, "tensorflow7-time", execute_tf)

        results_file = "%stime_test_%s_ll_%s.txt" % (outprefix, "tensorflow3", OS_name)
        if not os.path.isfile(results_file):
            ll, test_time = execute_tf()
            print("mean ll", np.mean(ll))
            np.savetxt(results_file, ll, delimiter=";")

        nfile = outprefix + "spnexe_" + OS_name

        def execute_native():

            print("computing ll for: ", exp, test_data.shape, nfile)
            cmd = "%s < %s" % (nfile, test_data_fname)
            proc_output = subprocess.check_output(cmd, shell=True).decode("utf-8")
            print("done")
            lines = proc_output.split("\n")
            cpp_ll = np.array(lines[0 : test_data.shape[0]], dtype=np.float128)
            cpp_time = float(lines[-2].split(" ")[-2])

            return cpp_ll, cpp_time

        run_experiment(exp, spn, test_data, "native", execute_native)

        nfile = outprefix + "spnexe_" + OS_name + "_fastmath"
        run_experiment(exp, spn, test_data, "native_fast", execute_native)

        def execute_python():
            start = time.perf_counter()
            py_ll = likelihood(spn, test_data)
            end = time.perf_counter()
            elapsed = end - start

            return py_ll, elapsed * 1000000000

        run_experiment(exp, spn, test_data, "python", execute_python)
