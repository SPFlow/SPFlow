'''
Created on March 29, 2018

@author: Alejandro Molina
'''
from spn.leaves.Histograms import Histogram_str_to_spn

from spn.experiments.FPGA.GenerateSPNs import fpga_count_ops
from spn.io.Graphics import plot_spn2
from spn.io.Text import str_to_spn

if __name__ == '__main__':
    with open('../experiments/FPGA/spns/NIPS_30/eqq.txt', 'r') as myfile:
        eq = myfile.read()
    with open('../experiments/FPGA/spns/NIPS_30/all_data.txt', 'r') as myfile:
        words = myfile.readline().strip()
        words = words[2:]
        words = words.split(';')

    print(words)

    spn = str_to_spn(eq, words, Histogram_str_to_spn)

    print(spn)

    print(fpga_count_ops(spn))
    plot_spn2(spn)


