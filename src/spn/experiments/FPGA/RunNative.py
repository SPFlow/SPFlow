'''
Created on March 26, 2018

@author: Alejandro Molina
'''
import glob
import os
import subprocess
import platform
import numpy as np

from spn.algorithms.Inference import log_likelihood
from spn.experiments.FPGA.GenerateSPNs import load_spn_from_file
from spn.leaves.Histograms import Histogram_Likelihoods
from spn.structure.Base import get_nodes_by_type, Node, get_number_of_edges, get_number_of_layers, Product, Leaf, Sum
from natsort import natsorted

np.set_printoptions(precision=50)



if __name__ == '__main__':
    path = os.path.dirname(__file__)

    for exp in natsorted(map(os.path.basename, glob.glob(path + '/spns/*'))):
        print(exp)
        ds_name, top_n_features = exp.split("_")
        top_n_features = int(top_n_features)

        outprefix = path + "/spns/%s_%s/" % (ds_name, top_n_features)

        py_ll = np.load(outprefix + "time_test_ll.npy")

        OS_name = platform.system()

        nfile = outprefix + "spnexe_" + OS_name

        spn, words, _ = load_spn_from_file(outprefix)

        data = np.loadtxt(outprefix + "all_data.txt", delimiter=";")

        r = np.random.RandomState(17)
        time_test = data[r.choice(data.shape[0], 10000), :]

        np.savetxt(outprefix + "time_test_data.txt", time_test, delimiter=";", header=";".join(words))

        py_ll = log_likelihood(spn, time_test, Histogram_Likelihoods)
        np.savetxt(outprefix + "time_test_ll.txt", py_ll)
        np.save(outprefix + "time_test_ll.npy", py_ll)

        print("computing ll natively for: ", outprefix, time_test.shape)
        cmd = "%s < %s" % (nfile, outprefix + "time_test_data.txt")
        proc_output = subprocess.check_output(cmd, shell=True).decode("utf-8")
        print("done")

        lines = proc_output.split("\n")
        cpp_ll = np.array(lines[0:time_test.shape[0]], dtype=np.float128)
        cpp_time = float(lines[-2].split(" ")[-2])

        native_ll_file = open(outprefix + "time_test_native_ll.txt", "w")
        native_ll_file.write("\n".join(lines[0:time_test.shape[0]]))
        native_ll_file.close()

        test_n = time_test.shape[0]

        if not os.path.isfile("results.csv"):
            results_file = open("results.csv", "w")
            results_file.write(";".join(
                ["Experiment", "OS", "machine", "input_rows", "input_cols", "spn_nodes", "spn_sum_nodes", "spn_prod_nodes", "spn_leaves", "spn_edges", "spn_layers", "time per instance", "time per task", "avg ll native", "avg ll pyspn"]))
            results_file.write("\n")
            results_file.close()

        import cpuinfo

        machine = cpuinfo.get_cpu_info()["brand"]

        results_file = open("results.csv", "a")
        results_file.write(";".join([exp, OS_name, machine]))
        results_file.write(";")
        results_file.write(";".join(map(str, [time_test.shape[0], time_test.shape[1]])))
        results_file.write(";")
        spn_stats = [len(get_nodes_by_type(spn, Node)),
                     len(get_nodes_by_type(spn, Sum)),
                     len(get_nodes_by_type(spn, Product)),
                     len(get_nodes_by_type(spn, Leaf)),
                     get_number_of_edges(spn),
                     get_number_of_layers(spn)]
        results_file.write(";".join(map(str, spn_stats)))
        results_file.write(";")
        results_file.write(";".join(map(str, [cpp_time / test_n, cpp_time])))
        results_file.write(";")
        results_file.write(";".join(map(str, [np.mean(cpp_ll, dtype=np.float128), np.mean(py_ll, dtype=np.float128)])))
        results_file.write("\n")
        results_file.close()
