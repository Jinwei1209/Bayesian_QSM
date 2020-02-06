import numpy as np

# Plot
def plots(ims, figsize=(12,6), 
               rows=1, 
               scale=None, 
               interp=False, 
               titles=None):
    from matplotlib import pyplot as plt
    
    if scale != None:
        lo, hi = scale
        ims = ims.copy()
        ims[ims > hi] = hi
        ims[ims < lo] = lo
        ims = (ims - lo)/(hi - lo) * 1.0
        
    if ims.ndim == 2:
        ims = ims[np.newaxis, ..., np.newaxis];
    elif ims.ndim == 3:
        ims = ims[..., np.newaxis];
    ims = np.tile(ims, (1,1,1,3))
    #ims = ims.astype(np.uint8)
    #print(ims.shape)
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows if len(ims) % 2 == 0 else len(ims)//rows + 1
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None:
            sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], interpolation=None if interp else 'none')

# Data processing
def resize_in_plane(inp):
    size = inp.shape
    out = np.zeros([size[0]//2, size[1]//2, size[2]])
    for i in range(0, size[0]//2):
        for j in range(0, size[1]//2):
            out[i, j, :] = (inp[2*i, 2*j, :] + inp[2*i+1, 2*j, :] + inp[2*i, 2*j+1, :] + inp[2*i+1, 2*j+1, :])/4
    return out

def extract_patches(img, patch_shape, extraction_step):
    from sklearn.feature_extraction.image import extract_patches as sk_extract_patches
    patches = sk_extract_patches(img, patch_shape=patch_shape, extraction_step=extraction_step)
    ndim = img.ndim
    npatches = np.prod(patches.shape[:ndim])
    return patches.reshape((npatches, ) + patch_shape)

# in-plane roration
def rotate_xy(img, theta, voxel_size=(1,1,1)):
    '''
    In-plane (x-y) rotation by {theta} degree
    '''
    from scipy.interpolate import RegularGridInterpolator
    theta = theta / 180 * np.pi
    matrix_size = img.shape
    rot = np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])
    FOV = np.array(matrix_size) * np.array(voxel_size)
    gridX = np.linspace(-0.5, 0.5, matrix_size[1])*FOV[1]
    gridY = np.linspace(-0.5, 0.5, matrix_size[0])*FOV[0]
    gridZ = np.linspace(-0.5, 0.5, matrix_size[2])*FOV[2]
    [X, Y, Z] = np.meshgrid(gridX, gridY, gridZ)
    loc = np.stack((Y.flatten(), X.flatten(), Z.flatten()), axis=1)
    loc_rot = np.dot(loc, rot.T);
    interp3 = RegularGridInterpolator((gridY, gridX, gridZ), img, method='nearest', bounds_error=False, fill_value=0)
    img_rot = interp3(loc_rot).reshape(matrix_size)
    return img_rot

# data augmentation (flip and rotation)
def augment_data(img, voxel_size=(1,1,1),
                      flip='xyz',
                      thetas=[]):
    '''
    Augment img by flipping and/or rotation
    Input:
        img,            ndarray(nx, ny, nz)
        voxel_size,     Tuple[int]
        flip,           str,                        axis for flipping
        thetas,         List[float],                angles for rotation (x-y plane)
    Output:
        imgs_aug,       ndarray(nsample, nx, ny, nz)
    '''
    imgs_aug = []
    imgs_aug.append(img)
    if 'x' in flip:
        imgs_aug.append(np.flip(img, axis=0))
    if 'y' in flip:
        imgs_aug.append(np.flip(img, axis=1))
    if 'z' in flip:
        imgs_aug.append(np.flip(img, axis=2))
    for theta in thetas:
        imgs_aug.append(rotate_xy(img, theta, voxel_size=voxel_size))
    imgs_aug = np.stack(imgs_aug, axis=0)
    return imgs_aug

# Data reconstruction
def generate_indexes(img_size, patch_shape, extraction_step):
    import itertools
    ndims = len(patch_shape)
    # Patch center template
    #   [starts[i]                                                              starts[i]+patch_shape[i])
    #         [starts[i]+bound[i]        starts[i]+bound[i]+extraction_step[i])
    #   bound[i] = (patch_shape[i] - extraction_step[i])//2
    bound = [ (patch_shape[i] - extraction_step[i])//2 for i in range(ndims) ]
    npatches = [ (img_size[i] - patch_shape[i])//extraction_step[i] + 1 for i in range(ndims) ]
    
    starts = [ list(range(0, npatches[i]*extraction_step[i], extraction_step[i])) for i in range(ndims) ]
    ends = [ [ start+patch_shape[i] for start in starts[i] ] for i in range(ndims) ]
    starts_bound = [ [ start+bound[i] for start in starts[i] ] for i in range(ndims) ]
    ends_bound = [ [ start+bound[i]+extraction_step[i] for start in starts[i] ] for i in range(ndims) ]
    starts_local = [ [ bound[i] for start in starts[i] ] for i in range(ndims) ]
    ends_local = [ [ bound[i]+extraction_step[i] for start in starts[i] ] for i in range(ndims) ]
    
    # Extend to the edge of image
    for i in range(ndims):
        starts_bound[i][0] = 0
        starts_local[i][0] = 0
        ends_bound[i][-1] = ends[i][-1]
        ends_local[i][-1] = patch_shape[i]
    
    idxs = [ list(zip(starts_bound[i], ends_bound[i], starts_local[i], ends_local[i])) for i in range(ndims)]
    return itertools.product(*idxs)

def reconstruct_patches(patches, img_size, extraction_step, idxs_valid=None):
    npatches, *patch_shape = patches.shape
    if idxs_valid is None:
        idxs_valid = [True] * npatches
    count_valid = 0
    reconstructed_img = np.zeros(img_size, dtype=patches.dtype)
    for count, idx in enumerate(generate_indexes(img_size, patch_shape, extraction_step)):
        start_bound, end_bound, start_local, end_local = zip(*list(idx))
        selection_bound = [slice(start_bound[i], end_bound[i]) for i in range(len(idx))]
        selection_local = [slice(start_local[i], end_local[i]) for i in range(len(idx))]
        if idxs_valid[count]:
            reconstructed_img[tuple(selection_bound)] = patches[count_valid][tuple(selection_local)]
            count_valid += 1
    return reconstructed_img