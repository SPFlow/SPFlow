'''
Created on March 20, 2018

@author: Alejandro Molina
'''


class Sum:
    def __init__(self):
        self.weights = []
        self.scope = []
        self.children = []


class Product:
    def __init__(self):
        self.scope = []
        self.children = []


class Leaf:
    def __init__(self):
        self.scope = []


class SPN:
    def __init__(self):
        self.root = None
        self.config = {}

class Context:
    pass