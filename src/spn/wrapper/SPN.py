'''
Created on May 15, 2018

@author: Moritz
'''

import numpy as np

from spn.io import Graphics
from spn.algorithms import Sampling
from spn.algorithms import Inference

class SimpleSPN(object):



    def __init__(self, root_node, feature_types, ds_context, inference_support, inference_support_ranges):
        self.root_node = root_node
        self.feature_types = feature_types
        self.ds_context = ds_context
        self.inference_support = inference_support
        self.inference_support_ranges = inference_support_ranges
    
   
    
    def eval_single(self, instance):
        instances = np.array([instance])
        return Inference.likelihood(self.root_node, instances, np.float64, self.ds_context, self.inference_support)[0][0]
    
    def eval_multi(self, instances):
        return Inference.likelihood(self.root_node, instances, np.float64, self.ds_context, self.inference_support)[:,0]
        
    def eval_dist(self, instance, featureIdx):
        return Inference.likelihood_dists(self.root_node,  instance, featureIdx, np.float64, self.ds_context, self.inference_support)
   
    
    
    def eval_ranges_single(self, rang):
        ranges = np.array([rang])
        return Inference.likelihood(self.root_node, ranges, np.float64, self.ds_context, self.inference_support_ranges)[0][0]
        
    def eval_ranges_multi(self, ranges):
        return Inference.likelihood(self.root_node, ranges, np.float64, self.ds_context, self.inference_support_ranges)[:,0]

    def eval_ranges_dist(self, rang, featureIdx):
        return Inference.likelihood_dists(self.root_node, rang, featureIdx, np.float64, self.ds_context, self.inference_support_ranges)
    
   
   
    def sample_random_new(self, n_instances, rand_gen):
        return Sampling.sample_instances(self.root_node, len(self.feature_types), n_instances, rand_gen)[0]
    
    def sample_evidence_new(self):
        ''' TODO '''
        pass
    
    def sample_ranges_new(self):
        ''' TODO '''
        pass
    
    
    
    def sample_random_old(self, n_instances, rand_gen):
        ''' TODO '''
        pass
        
    def sample_evidence_old(self):
        ''' TODO '''
        pass

    def sample_ranges_old(self):
        ''' TODO '''
        pass
    
    

    def to_svg(self, file_name="plot.svg"):        
        Graphics.plot_spn_to_svg(self.root_node, file_name)



if __name__ == '__main__':
    
    '''
    Simple example how the wrappper-class can be used!
    '''
    
    
    '''    Modify if necessary    '''
    import os 
    os.environ["R_HOME"] = "C:/Program Files/R/R-3.3.3" 
    os.environ["R_USER"] = "Moritz" 
    
    from spn.structure.Base import Context
    from spn.structure.StatisticalTypes import MetaType
    from spn.structure.Ranges import NominalRange
    
    from spn.structure.leaves.piecewise.PiecewiseLinear import PiecewiseLinear
    from spn.structure.leaves.piecewise.Inference import piecewise_log_likelihood
    from spn.structure.leaves.piecewise.Inference import piecewise_log_likelihood_range
    
    from spn.structure.leaves.parametric.Parametric import Categorical
    from spn.structure.leaves.parametric.Inference import parametric_log_likelihood
    from spn.structure.leaves.parametric.Inference import parametric_log_likelihood_range
    
    #Create SPN
    node1 = Categorical(p=[0.1, 0.9], scope=[0])
    node2 = Categorical(p=[0.5, 0.5], scope=[0])
    root_node = 0.4 * node1 +  0.6 * node2
    
    #Set feature types
    feature_types = ["discrete"]
    
    #Set context
    meta_types = [MetaType.DISCRETE]
    domains = [[0,1]]
    ds_context = Context(meta_types=meta_types, domains=domains)
    
    #Set support for inference
    inference_support =        {PiecewiseLinear : piecewise_log_likelihood, 
                                Categorical     : parametric_log_likelihood}
    inference_support_ranges = {PiecewiseLinear : piecewise_log_likelihood_range, 
                                Categorical     : parametric_log_likelihood_range}
    
    #Create wrapper
    spn = SimpleSPN(root_node, feature_types, ds_context, inference_support, inference_support_ranges)
    
    
    
    #Create a svg of the spn
    spn.to_svg("simple_spn_example.svg")
    


    #Evaluate a single instance
    instance = np.array([0])
    prob = spn.eval_single(instance)
    print("Evaluate a single instance: " + str(prob))
    
    
    #Evaluate  multiple instances
    instances = np.array([[0],
                         [1]])
    probs = spn.eval_multi(instances)
    print("Evaluate  multiple instances: " + str(probs))
    
    
    #Evaluate  multiple instances range
    instances = np.array([[None],
                         [NominalRange([0])],
                         [NominalRange([1])],
                         [NominalRange([0,1])]])
    probs = spn.eval_ranges_multi(instances)
    print("Evaluate  multiple instances range: " + str(probs))
    
    
    #Generate samples
    rand_gen = np.random.RandomState(100)
    samples = spn.sample_random_new(5, rand_gen)
    print("Sample instances:")
    print(samples)
