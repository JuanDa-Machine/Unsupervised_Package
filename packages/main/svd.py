
import numpy as np

class SVD:

    def __init__(self,n_vectors):   
        self.n_vectors=n_vectors 

    def fit(self,x):
        '''Creates the matrtixes for SVD transformation and generates the truncate matrix, which allows
        to reduce new features using
        params used:
        x: Data to train
        n_vectors: How many vectors you will use
        ''' 
        self.x=x  
        #compute the vectors
        self.U, self.s, self.Vt = np.linalg.svd(self.x) 
        #take the n_components we need
        self.Uk = self.U[:, :self.n_vectors]
        self.sk = np.diag(self.s[:self.n_vectors])
        self.Vk = self.Vt[:self.n_vectors, :]    
        #compute mean and std to standarization    
        self.mu = np.mean(self.x, axis=0)
        self.sigma = np.std(self.x, axis=0)
        #compute truncate svd
        # self.truncate_svd = (Uk@ np.diag(sk)@ Vk)
        self.truncate_svd=self.Vk.T

    def transform(self,x):   
        # X_new_centered = x - self.mu
        # X_new_scaled = X_new_centered / self.sigma
        # return np.dot(X_new_scaled, self.truncate_svd)
        return ( x@ self.truncate_svd)
    
    def fit_transform(self,x):
        self.x=x  
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self): 
        X_reconstructed = ( self.x@ self.truncate_svd).dot(self.truncate_svd.T) + np.mean(self.x, axis=0)      
        return X_reconstructed