import numpy as np

def compute_Z(X, centering=True, scaling=False):
    Z = np.copy(X)
    if centering:
        Z = Z - Z.mean(axis=0)
    if scaling:
        Z = Z / Z.std(axis=0)
    return Z 

def compute_covariance_matrix(Z):
    return np.matmul(Z.T, Z)

def find_pcs(COV):
    eigVal, eigVec = np.linalg.eig(COV)
    sortIndex = np.argsort(eigVal)[::-1]
    return eigVal[sortIndex], eigVec[sortIndex]

def project_data(Z, PCS, L, k, var):
    if k == 0:
        for _ in L:
            k += 1        
            expVar = L[:k].sum() / L.sum()
            if  expVar >= var:
                break
    return np.array([np.matmul(Z, PCS[:,i]) for i in range(k)]).T
