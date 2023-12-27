import numpy as np
from Analysis import Analysis
import glob
from tqdm import tqdm
def flatten(t):
    return [item for sublist in t for item in sublist]

def mask(x):
    'This function is very specific for this analysis'
    for i in range((x.shape[0])):
        for j in range((x.shape[1])):
            if np.isneginf(x[i][j][3]):
                x[i][j] = [0,0,0,0,0,0,0]
    return x 
    
    
def prepare_files (dir_,n):
    files = glob.glob(dir_+'/*')
    cont_pt,cont_eta,cont_phi,cont_E = [],[],[],[]
    kin_sig=np.empty(shape=[0,12])
    for file in files:
        S = Analysis(file,200,n)
        x = S.Analysis_reco()
        kin_sig = np.append(kin_sig,np.array(x[:12]).T,axis=0)
        cont_pt.append(x[12]);cont_eta.append(x[13])
        cont_phi.append(x[14])
        cont_E.append(x[15])    
    'returns: particle contents + kinematic jet distributions'    
    return cont_pt,cont_eta,cont_phi,cont_E,kin_sig    
 
 
def rot_phi(cont_phi):
    '''Normaly particles can scatter anywhere in the phi
    direction of the detector. This function uses the symetry in phi
    constrain phi between 0 and 2pi '''
    for i in tqdm(range(cont_phi.shape[0])):
        phit = cont_phi[i,:] - cont_phi[i,0]
        cont_phi[i,phit < -np.pi] += 2*np.pi
        cont_phi[i,phit > np.pi]  += -2*np.pi 
    return cont_phi
    
    
def centroid(x,y,weight,x_power,y_power):
    return ((x**x_power)*(y**y_power)*weight).sum()
    
def centre_jet(x, y, weights):
    x_centre = centroid(x,y, weights, 1,0) / weights.sum()
    y_centre = centroid(x,y, weights,0, 1)/ weights.sum()
    x = x - x_centre
    y = y - y_centre
    return x, y
    
# The used roation function based on Heidelberg code
def rotate_jet(x, y, weights):
      u11 = centroid(x, y, weights, 1, 1) / weights.sum()
      u20 = centroid(x, y, weights, 2, 0) / weights.sum()
      u02 = centroid(x, y, weights, 0, 2) / weights.sum()
      cov = np.array([[u20, u11], [u11, u02]])
      # Eigenvalues and eigenvectors of covariant matrix
      evals, evecs = np.linalg.eig(cov)
      # Sorts the eigenvalues, v1, [::-1] turns array around, 
      sort_indices = np.argsort(evals)[::-1]
      e_1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
      e_2 = evecs[:, sort_indices[1]]
      # Theta to x_axis, arctan2 gives correct angle
      theta = np.arctan2(e_1[0], e_2[0])
      # Rotation, so that princple axis is vertical
      # Anti-clockwise rotation matrix
      rotation = np.matrix([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
      transformed_mat = rotation * np.stack([x,y])
      x_rot, y_rot = transformed_mat.A # To return an array instead of a matrix
      
      return x_rot, y_rot            
      
def flip_jet(x, y, weights):
    if weights[x<0.].sum() < weights[x>0.].sum():
        x = -x
    if weights[y<0.].sum() > weights[y>0.].sum():
        y = -y
    return x,y 
    
def orig_image (etas, phis, es):
    # Grid settings
    xpixels = np.arange(-2.6, 2.6, 0.029)
    ypixels = np.arange(-np.pi, np.pi, 0.035)
    # gives the value on grid with minimal distance,
    # eg. for xpixel = (0,1,2,3,..) eta=1.3 -> xpixel=1, eta=1.6 ->xpixel=2
    # first define the grid full of zeros
    z = np.zeros( ( etas.shape[0], len(xpixels), len(ypixels) ) )
    # now we want a vector telling us if a point is in the grid
    in_grid = ~((etas < xpixels[0]) | (etas > xpixels[-1]) | (phis < ypixels[0]) | (phis > ypixels[-1]))
    # for each eta and phi we now find the cell on the grid that is closest to the actual coordinate
    xcoords = np.argmin( np.abs( etas[:,None,:] - xpixels[None,:,None] ), axis=1 )
    ycoords = np.argmin( np.abs( phis[:,None,:] - ypixels[None,:,None] ), axis=1 )
    # create a grid where each row is filled with the event number
    ncoords = np.repeat( np.arange( etas.shape[0])[:,None], etas.shape[1], axis=1 )
    # create a single array for all images, for each jet there is a 180x180 matrix
    z[ ncoords[in_grid], ycoords[in_grid], xcoords[in_grid] ] = es[in_grid]
    return z    
