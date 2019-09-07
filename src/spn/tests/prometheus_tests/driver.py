from spn.algorithms.Inference import log_likelihood
from scipy.cluster.vq import whiten
import tensorflow as tf
from spn.io.Text import spn_to_str_equation, str_to_spn
from spn.gpu.TensorFlow import optimize_tf
import sys
import numpy as np

sys.setrecursionlimit(15000)

with open('./ca.txt', 'r') as myfile:
    data = myfile.read().replace('\n', '')

thespn = str_to_spn(data)

testdata = np.load('./test.npy')

testdata += 0.0

print(testdata.dtype)
print(np.amax(testdata))
#testdata = whiten(testdata)
print(np.amax(testdata))

testdata = testdata.astype(np.float32)

print(testdata.dtype)

ll = log_likelihood(thespn, testdata)
print(ll, np.exp(ll))

optimized_spn = optimize_tf(
    thespn,
    testdata,
    epochs=100,
    optimizer=tf.train.RMSPropOptimizer(1e-4))
lloptimized = log_likelihood(optimized_spn, testdata)
print(lloptimized, np.exp(lloptimized))
print(np.mean(lloptimized))
print(np.mean(ll))

txt = spn_to_str_equation(optimized_spn)

'''
text_file = open("./optca.txt", "w")
text_file.write(txt)
text_file.close()
'''
