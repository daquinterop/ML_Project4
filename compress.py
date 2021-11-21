import numpy as np
from matplotlib import pyplot as plt
import pca
import os

def compress_images(DATA,k):
    if os.path.exists('Output'):
        for i in os.listdir('Output'): os.remove(os.path.join('Output', i))
        os.rmdir('Output')
    os.mkdir('Output')
    Z = pca.compute_Z(DATA, centering=True, scaling=True)
    Z_cov = pca.compute_covariance_matrix(Z)
    L, PCS = pca.find_pcs(Z_cov)
    Z_star = pca.project_data(Z, PCS, L, k, None)
    X_comp = np.matmul(Z_star, PCS[:,:k].T)
    X_comp = (255 * (X_comp - X_comp.min())/(X_comp.max() - X_comp.min())).astype('uint8')
    for n, image in enumerate(X_comp.T):
        plt.imsave(fname=os.path.join('Output', f'{n:03d}.png'), arr=image.reshape((60, 48)), cmap='gray', format='png')
    return 


def load_data(input_dir):
    # np.fromfile(os.path.join(input_dir, os.listdir(input_dir)[0]), dtype=np.int8).shape
    data = []
    for filename in os.listdir(input_dir):
        try:
            image = plt.imread(os.path.join(input_dir, filename))
            data.append(image.flatten())
        except TypeError:
            continue
    data = np.array(data).astype(float).T
    return data
