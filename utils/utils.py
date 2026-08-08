import torch
import math
import numpy as np
from pathlib import Path
import torch.nn.functional as F
import torchvision.transforms as T

def init_seeds(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def add_latent(points, latent_codes):
    batch_size, num_of_points, dim = points.shape
    points = points.reshape(batch_size * num_of_points, dim)
    latent_codes = latent_codes.reshape(batch_size * num_of_points, -1)
    out = torch.cat([latent_codes, points], 1)
    
    return out

def save_obj_data(filename, vertex, face):
	numver = vertex.shape[0]
	numfac = face.shape[0]
	with open(filename, 'w') as f:
		f.write('# %d vertices, %d faces'%(numver, numfac))
		f.write('\n')
		for v in vertex:
			f.write('v %f %f %f' %(v[0], v[1], v[2]))
			f.write('\n')
		for F in face:
			f.write('f %d %d %d' %(F[0]+1, F[1]+1, F[2]+1))
			f.write('\n')