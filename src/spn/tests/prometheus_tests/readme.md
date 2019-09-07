Run test.py to generate a structure using prometheus from test.npy.

Run driver.py to test that RMSProp correctly optimizes this structure. Note that the cholesky error does not occur due to using MVGTril over MVGFull. If done correctly, in the whole array of lls, no nans should exist.

Known bug: A high number of iters with Adam or RMSProp with a high learning rate will take the model to a point where inference will break the no nan rule. I do not know why.
