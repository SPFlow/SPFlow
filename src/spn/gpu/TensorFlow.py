'''
Created on March 27, 2018

@author: Alejandro Molina
'''


def get_tf_graph(node):
    self.childrenprob = tf.stack([c.value for c in self.children], axis=1)
    # self.value = tf.reduce_prod(self.childrenprob, 1) #in prob space
    self.value = tf.reduce_sum(self.childrenprob, 1)