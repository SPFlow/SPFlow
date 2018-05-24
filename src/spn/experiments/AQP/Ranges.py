'''
Created on May 22, 2018

@author: Moritz
'''


from abc import ABC, abstractmethod

import numpy as np

class Range(ABC):
    
    @abstractmethod
    def is_impossible(self):
        '''
        Returns true if the range cannot be fulfilled.
        '''
        pass
    
    @abstractmethod
    def get_ranges(self):
        '''
        Returns the values for the ranges.
        '''
        pass
    


class NominalRange(Range):
    '''
    This class specifies the range for a nominal attribute. It contains a list of integers which
    represent the values which are in the range.
    
    e.g. possible_values = [5,2] 
    '''
    
    def __init__(self, possible_values):
        self.possible_values= np.array(possible_values, dtype=np.int64)
        
    def is_impossible(self):
        return len(self.possible_values) == 0
    
    def get_ranges(self):
        return self.possible_values

        
        
class NumericRange(Range):
    '''
    This class specifies the range for a numeric attribute. It contains a list of (closed) intervals which
    represents the values which are valid.
    
    e.g. ranges = [[10,15],[22,23]] if valid values are between 10 and 15 plus 22 and 23 (bounds inclusive)
    '''
    
    def __init__(self, ranges):
        self.ranges = ranges

    def is_impossible(self):
        return len(self.ranges) == 0

    def get_ranges(self):
        return self.ranges
