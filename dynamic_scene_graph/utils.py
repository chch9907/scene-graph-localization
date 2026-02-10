import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def to_tensor(inputs, dtype='torch.float32', device='cuda'):
    if isinstance(inputs, np.ndarray):
        return torch.from_numpy(inputs).to(device).float()
    elif isinstance(inputs, list):
        return torch.tensor(inputs).to(device).float()
    else:
        return inputs.to(device).float()
    
def get_center_dist(ct1, ct2):
    '''can be used to calculate 2D or 3D distance'''
    if not isinstance(ct1, np.ndarray):
        ct1 = np.array(ct1)
    if not isinstance(ct2, np.ndarray):
        ct2 = np.array(ct2)  
    return np.linalg.norm(ct1 - ct2)

def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()