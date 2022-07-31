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
                          BinaryClassificationDistribution,
                          UniformCapDistribution
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
        
        
        density1 = UniformDistribution(lower=0,upper=0.5) 
        density2 = UniformDistribution(lower=0.5,upper=1) 
        density_vec=[density1, density2]
        mixed_density_1=MixedDistribution(density_vec,[0.2,0.8])
        mixed_density_2=MixedDistribution(density_vec,[0.8,0.2])
        density_class_1=MarginalDistribution([mixed_density_1 for _ in range(dim)])
        density_class_2=MarginalDistribution([mixed_density_2 for _ in range(dim)])
    
        density_seq = [density_class_1, density_class_2]
        prob_seq = [0.5,0.5]
        
        densityBinary = BinaryClassificationDistribution(density_seq, prob_seq)
        return densityBinary
    
    def testDistribution_4(self,dim):
        
        density1 = UniformCapDistribution(propotion=0.1,dim=dim,postive_direction=True)
        density2 = UniformCapDistribution(propotion=0.1,dim=dim,postive_direction=False)
        
    
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
    
