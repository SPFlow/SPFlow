'''
Created on March 26, 2018

@author: Alejandro Molina
'''
import os
import glob

from natsort import natsorted

from spn.experiments.FPGA.GenerateSPNs import load_spn_from_file
from spn.io.CPP import generate_native_executable
from spn.leaves.Histograms import Histogram_to_cpp

if __name__ == '__main__':

    path = os.path.dirname(__file__)

    for exp in natsorted(map(os.path.basename, glob.glob(path+'/spns/*'))):
        print(exp)
        ds_name, top_n_features = exp.split("_")
        top_n_features = int(top_n_features)

        outprefix = path + "/spns/%s_%s/" % (ds_name, top_n_features)

        spn, words, _ = load_spn_from_file(outprefix)

        import platform

        nfile = outprefix + "spnexe_" + platform.system()

        compilation_results = generate_native_executable(spn, Histogram_to_cpp,
                                                         cppfile=outprefix + "spn.cpp", nativefile=nfile)

        print(compilation_results[0], compilation_results[1])

