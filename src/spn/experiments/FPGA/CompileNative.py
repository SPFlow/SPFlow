"""
Created on March 26, 2018

@author: Alejandro Molina
"""
import glob
import os

from natsort import natsorted

from spn.experiments.FPGA.GenerateSPNs import load_spn_from_file
from spn.io.CPP import generate_native_executable

if __name__ == "__main__":

    path = os.path.dirname(__file__)

    for exp in natsorted(map(os.path.basename, glob.glob(path + "/spns/*"))):
        print(exp)

        outprefix = path + "/spns/%s/" % (exp)

        import platform

        nfile = outprefix + "spnexe_" + platform.system()

        if os.path.isfile(nfile):
            continue

        spn, words, _ = load_spn_from_file(outprefix)

        cpp_file = outprefix + "spn.cpp"
        compilation_results = generate_native_executable(spn, cppfile=cpp_file, nativefile=nfile)

        print(compilation_results[0], compilation_results[1])
