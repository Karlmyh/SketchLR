import numpy as np
import math
from sklearn.utils import check_random_state
from numba import njit
from scipy.linalg import hadamard




__all__ = ['GaussianSketch', 'CountSketch', 'SubsamplingSketch',"SRHT"]

@njit
def gaussian_apply(matrix,sketch_matrix,apply_left,apply_right):
    if apply_left:
        matrix=  sketch_matrix @ matrix
    if apply_right:   
        transpose_sketch_matrix=np.ascontiguousarray(sketch_matrix.T)
        matrix=  matrix @ transpose_sketch_matrix
    return matrix

@njit
def count_apply(matrix,
                sketch_dimension,
                rademacher,
                hashed_index,
                apply_left,
                apply_right):
    
    if apply_left:
        matrix_save=np.zeros((sketch_dimension,matrix.shape[1]))
        matrix_flip = rademacher.reshape(-1, 1) * matrix
        for i in range(sketch_dimension):
            idx=  (hashed_index == i)
            matrix_save[i, :] = np.sum(matrix_flip[idx, :], 0)
        matrix=matrix_save
              
    if apply_right:
        matrix_save=np.zeros((matrix.shape[0],sketch_dimension))
        matrix_flip = matrix * rademacher
        for i in range(sketch_dimension):
            idx = (hashed_index == i)
            matrix_save[:, i] = np.sum(matrix_flip[:, idx], 1)
        matrix=matrix_save
    return matrix
    
@njit
def subsampling_apply(matrix,
                dimension_ratio,
                hashed_index,
                apply_left,
                apply_right):
    
    if apply_left:
        matrix=matrix[hashed_index,:]*math.sqrt(dimension_ratio)
              
    if apply_right:
        matrix=matrix[:,hashed_index]*math.sqrt(dimension_ratio)
        
    return matrix



@njit
def srht_apply(matrix,
        sketch_dimension,
        rademacher,
        hashed_index,
        hadamard_matrix,
        apply_left,
        apply_right):
    
    if apply_left:
        matrix = 1/math.sqrt(sketch_dimension)*rademacher.reshape(-1, 1) * matrix
        matrix = hadamard_matrix.T @ matrix
        matrix=matrix[hashed_index,:]
              
    if apply_right:
        
        matrix = matrix *(rademacher/math.sqrt(sketch_dimension))
        matrix =   matrix@ hadamard_matrix
        matrix=matrix[:,hashed_index]
        
    return matrix


    
    
    
class SketchClass:
    
    def __init__( self, random_state=1 ):
        '''
        random_state : int, RandomState instance or None, default=None
            The seed of the pseudo random number generator used to generate a
            uniform distribution. If int, random_state is the seed used by the
            random number generator; If RandomState instance, random_state is the
            random number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.
        '''
        self.random_state = check_random_state(random_state).get_state()
        
        
    def Matricize(self, input_dimension, sketch_dimension):
        
        # keep random_state
        I = np.eye(input_dimension)
        sketch_matrix = self.Apply(I,sketch_dimension=sketch_dimension)

        return sketch_matrix

        
class GaussianSketch(SketchClass):
    
    
    def Apply(self, matrix, 
              sketch_dimension="auto",
              apply_left  = True,
              apply_right = False):
        
        # check in which case the input matrix falls in 
        if len(matrix.shape)==1 and apply_right and not apply_left:
            matrix=matrix.reshape(1,-1)
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_right and not apply_left:
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_left and not apply_right:
            original_dimension=matrix.shape[0]
        elif matrix.shape[0]==matrix.shape[1] and apply_left and apply_right:
            original_dimension=matrix.shape[0]
        else:
            raise ValueError(
                "Invalid matrix shape {} and operation choice".format(matrix.shape)
            )
        
        # check sketch dimension
        if isinstance(sketch_dimension,int) and sketch_dimension>0:
            pass
        elif sketch_dimension=="auto":
            sketch_dimension=math.ceil(math.log(original_dimension))
        else:
            raise ValueError(
                "{} is not a valid sketch dimension".format(sketch_dimension)
            )
        
        
        temp_state = np.random.get_state()
        np.random.set_state(self.random_state)
        sketch_matrix=np.random.normal(0, 1/math.sqrt(sketch_dimension), (sketch_dimension,original_dimension))
        np.random.set_state(temp_state)
        
        return gaussian_apply(matrix,sketch_matrix,apply_left,apply_right)
        
        
class SubsamplingSketch(SketchClass):

    
    def Apply(self, matrix,
              sketch_dimension="auto",
              apply_left  = True,
              apply_right = False):
        
        # check in which case the input matrix falls in
        if len(matrix.shape)==1 and apply_right and not apply_left:
            matrix=matrix.reshape(1,-1)
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_right and not apply_left:
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_left and not apply_right:
            original_dimension=matrix.shape[0]
        elif matrix.shape[0]==matrix.shape[1] and apply_left and apply_right:
            original_dimension=matrix.shape[0]
        else:
            raise ValueError(
                "Invalid matrix shape {} and operation choice".format(matrix.shape)
            )
        
        # check sketch dimension
        if isinstance(sketch_dimension,int) and sketch_dimension>0:
            pass
        elif sketch_dimension=="auto":
            sketch_dimension=math.ceil(math.log(original_dimension))
        else:
            raise ValueError(
                "{} is not a valid sketch dimension".format(sketch_dimension)
            )
            
        temp_state = np.random.get_state()
        np.random.set_state(self.random_state)
        
        dimension_ratio=original_dimension/sketch_dimension
        hashed_index = np.random.choice(original_dimension, sketch_dimension, replace=False)
        
        np.random.set_state(temp_state)
        
        return subsampling_apply(matrix,
                dimension_ratio,
                hashed_index,
                apply_left,
                apply_right)

class CountSketch(SketchClass):
    '''
    Input:
        original data matrix                 : m by n matrix A
        target skeching dimention for rows    : s
        indicator if sketch matrix is needed: boolean returnSketchMatrix
        
    Output:
        result matrix                         : s by n matrix C
        skeching matrix(if required)        : s by m matrix S
    '''
    
    def Apply(self, matrix, 
              sketch_dimension="auto",
              apply_left  = True,
              apply_right = False):
              
        # check in which case the input matrix falls in
        if len(matrix.shape)==1 and apply_right and not apply_left:
            matrix=matrix.reshape(1,-1)
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_right and not apply_left:
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_left and not apply_right:
            original_dimension=matrix.shape[0]
        elif matrix.shape[0]==matrix.shape[1] and apply_left and apply_right:
            original_dimension=matrix.shape[0]
        else:
            raise ValueError(
                "Invalid matrix shape {} and operation choice".format(matrix.shape)
            )
        
        # check sketch dimension
        if isinstance(sketch_dimension,int) and sketch_dimension>0:
            pass
        elif sketch_dimension=="auto":
            sketch_dimension=math.ceil(math.log(original_dimension))
        else:
            raise ValueError(
                "{} is not a valid sketch dimension".format(sketch_dimension)
            )
            
        temp_state = np.random.get_state()
        np.random.set_state(self.random_state)
        
        hashed_index = np.random.choice(sketch_dimension, original_dimension, replace=True)
        rademacher = np.random.choice(2, original_dimension, replace=True) * 2 - 1 
        np.random.set_state(temp_state)
        
        return count_apply(matrix,
                sketch_dimension,
                rademacher,
                hashed_index,
                apply_left,
                apply_right)

        
class SRHT(SketchClass):

    def Apply(self, matrix, 
              sketch_dimension="auto",
              apply_left  = True,
              apply_right = False):
              
        # check in which case the input matrix falls in
        if len(matrix.shape)==1 and apply_right and not apply_left:
            matrix=matrix.reshape(1,-1)
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_right and not apply_left:
            original_dimension=matrix.shape[1]
        elif len(matrix.shape)==2 and apply_left and not apply_right:
            original_dimension=matrix.shape[0]
        elif matrix.shape[0]==matrix.shape[1] and apply_left and apply_right:
            original_dimension=matrix.shape[0]
        else:
            raise ValueError(
                "Invalid matrix shape {} and operation choice".format(matrix.shape)
            )
        
        # check sketch dimension
        if isinstance(sketch_dimension,int) and sketch_dimension>0:
            pass
        elif sketch_dimension=="auto":
            sketch_dimension=math.ceil(math.log(original_dimension))
        else:
            raise ValueError(
                "{} is not a valid sketch dimension".format(sketch_dimension)
            )
            
        temp_state = np.random.get_state()
        np.random.set_state(self.random_state)
        
        hashed_index = np.random.choice(original_dimension, sketch_dimension, replace=False)
        hadamard_matrix=hadamard(int(2**np.ceil(np.log2(original_dimension))))[:original_dimension,:original_dimension]
        
        rademacher = np.random.choice(2, original_dimension, replace=True) * 2 - 1 
        np.random.set_state(temp_state)
        
        return srht_apply(matrix,
                sketch_dimension,
                rademacher,
                hashed_index,
                hadamard_matrix,
                apply_left,
                apply_right)
    
