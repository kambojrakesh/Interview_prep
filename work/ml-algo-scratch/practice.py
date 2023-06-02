import numpy as np

# Define the PCA function
def pca(x, num_comp):
    x_sd = (x - np.mean(x, axis=0))/np.std(x, axis =0)
    cov = np.cov(x_sd.T)
    eva, eve = np.linalg.eig(cov)
    idx = np.argsort(eva)[::-1]
    
    matrix = eve[:,idx][:,:num_comp]
    
    return x_sd.dot(matrix) 




x = np.array([[1,2,1], [3,2,0], [5,1, 4]])
m = pca(x, 2)
print(m)