from .distributions import (LaplaceDistribution, 
                          BetaDistribution,
                          DeltaDistribution,
                          MultivariateNormalDistribution,
                          UniformDistribution,
                          MarginalDistribution,
                          ExponentialDistribution,
                          MixedDistribution,
                          UniformCircleDistribution,
                          CauchyDistribution,
                          CosineDistribution,
                          TDistribution,
                          BinaryClassificationDistribution
                          )
import numpy as np
import math


__all__ = ['TestDistribution']

class TestDistribution(object):
    def __init__(self,index,dim):
        self.dim=dim
        self.index=index
        
    def testDistribution_1(self,dim):
        
        density1 = MultivariateNormalDistribution(mean=np.zeros(dim)+1,cov=np.diag(np.ones(dim)*1)) 
        density2 = MultivariateNormalDistribution(mean=np.zeros(dim)-1,cov=np.diag(np.ones(dim)*1)) 
        
    
        density_seq = [density1, density2]
        prob_seq = [0.4,0.6]
        densityBinary = BinaryClassificationDistribution(density_seq, prob_seq)
        return densityBinary
    
    
    def testDistribution_2(self,dim):
        
        density1 = MultivariateNormalDistribution(mean=np.zeros(dim)+1,cov=np.diag(np.ones(dim)*1)) 
        density2 = MultivariateNormalDistribution(mean=np.zeros(dim)-1,cov=np.diag(np.ones(dim)*1)) 
        density_seq_1 = [density1, density2]
        prob_seq_1 = [0.4,0.6]
        densitymix = MixedDistribution(density_seq_1, prob_seq_1)
        
        density3=MultivariateNormalDistribution(mean=np.zeros(dim),cov=np.diag(np.ones(dim)*1)) 
    
        density_seq = [densitymix, density3]
        prob_seq = [0.5,0.5]
        
        densityBinary = BinaryClassificationDistribution(density_seq, prob_seq)
        return densityBinary
    
    
    def testDistribution_3(self,dim):
        
        density1 = UniformDistribution(lower=np.ones(dim)*0,upper=np.ones(dim)*0.5) 
        density2 = UniformDistribution(lower=np.ones(dim)*0.5,upper=np.ones(dim)*1) 
        
    
        density_seq = [density1, density2]
        prob_seq = [0.5,0.5]
        
        densityBinary = BinaryClassificationDistribution(density_seq, prob_seq)
        return densityBinary
    
    def testDistribution_4(self,dim):
        
        density1 = UniformDistribution(lower=np.ones(dim)*0,upper=np.ones(dim)*0.5) 
        density2 = UniformDistribution(lower=np.ones(dim)*0.4,upper=np.ones(dim)*1) 
        
    
        density_seq = [density1, density2]
        prob_seq = [0.5,0.5]
        
        densityBinary = BinaryClassificationDistribution(density_seq, prob_seq)
        return densityBinary
    
    
    def returnDistribution(self):
        switch = {'1': self.testDistribution_1,
                  '2': self.testDistribution_2,
                  '3': self.testDistribution_3,
                  '4': self.testDistribution_4,
                  }

        choice = str(self.index)             
        result=switch.get(choice)(self.dim)
        return result
    
