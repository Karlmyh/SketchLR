import numpy as np


class SketchGaussian:
    def __init__( self, inputdim, sketchdim,seed=1 ):
        self.inputdim 	= int(inputdim)
        self.sketchdim 	= int(sketchdim)
        self.seed=seed
        
        
    def Apply(self,A):
        np.random.seed(self.seed)
        S=np.random.normal(size=self.inputdim*self.sketchdim).reshape(self.sketchdim,self.inputdim)
        return np.matmul(S,A)