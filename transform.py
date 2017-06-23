'''
This module contains a collection of methods that transform and modify mask objects.
'''

from copy       import deepcopy
from image      import mask
import numpy    as np

def expand_surface(msk, expansion):
    '''
    Expand the binary mask surface by the given expansion in physical units (e.g., mm, cm, ...)

    Note that all expansions should be positive values!

    Positional arguments:
        :msk:       mask to transform
        :expansion: float or list of floats defining the expansion

    Note: Expansion may be given as:
        - A single number:       uniform expansion in all dimensions
        - A list of 1 number:    uniform expansion in all dimensions
        - A list of 3 numbers:   expansion in x, y, and z, respectively
        - A list of 6 numbers:   expansion in -x, +x, -y, +y, -z, and +z, respectively
    '''
    # Process the expansion. Convert to negative (n) and positive (p) x, y, and z
    if isinstance(expansion, str):
        expansion = float(expansion)

    try:
        n_exp = len(expansion)
    except:
        expansion = [expansion]
        n_exp = 1

    if any([e<0 for e in expansion]):
        raise(TypeError('Expansions and contractions cannot be negative'))

    if n_exp==1:
        nx = px = ny = py = nz = pz = float(expansion[0])
    elif n_exp==3:
        nx = px = float(expansion[0])
        ny = py = float(expansion[1])
        nz = pz = float(expansion[2])
    elif n_exp==6:
        nx, px, ny, py, nz, pz = [float(e) for e in expansion]
    else:
        raise(TypeError('Invalid expansion: '+str(expansion)))

    # Convert physical expansions to voxel expansions
    nix = int(round(nx / msk.spacing[0]))
    pix = int(round(px / msk.spacing[0]))
    niy = int(round(ny / msk.spacing[1]))
    piy = int(round(py / msk.spacing[1]))
    niz = int(round(nz / msk.spacing[2]))
    piz = int(round(pz / msk.spacing[2]))

    # Determine the indices of all neighboring voxels within the bounds of the voxel expansions
    shape = (niz+piz+1, niy+piy+1, nix+pix+1)
    kernel = np.zeros(shape, dtype=bool)
    for iz in range(-niz, piz+1):
        for iy in range(-niy, piy+1):
            for ix in range(-nix, pix+1):
                # For a voxel to be inside the expansion surface, do NOT assume
                # that the expansion surface must contain the voxel center. Instead,
                # assume that the expansion surface overlaps the ellipse positioned
                # at the center of the voxel having radii equal to 1/4 the voxel size.
                if ix < 0: x = nx + msk.spacing[0]/4
                else:      x = px + msk.spacing[0]/4
                if iy < 0: y = ny + msk.spacing[1]/4
                else:      y = py + msk.spacing[1]/4
                if iz < 2: z = nz + msk.spacing[2]/4
                else:      z = pz + msk.spacing[2]/4

                sum_of_squares = 0.0
                if x != 0: sum_of_squares += (ix * msk.spacing[0] / x)**2
                if y != 0: sum_of_squares += (iy * msk.spacing[1] / y)**2
                if z != 0: sum_of_squares += (iz * msk.spacing[2] / z)**2
                if sum_of_squares <= 1:
                    kernel[iz+niz, iy+niy, ix+nix] = 1

    expansion_voxels = np.transpose(np.nonzero(kernel))
    expansion_voxels[:,0] -= niz
    expansion_voxels[:,1] -= niy
    expansion_voxels[:,2] -= nix

    # Apply expansion voxels to mask surface
    expansion_mask = msk.get_mask_edge_voxels(exclude_z=False)
    expansion_mask_idx = np.transpose(np.nonzero(expansion_mask.data))
    for i in range(expansion_mask_idx.shape[0]):
        new_mask_idx = expansion_mask_idx[i,:] + expansion_voxels
        in_bounds = np.logical_and( new_mask_idx[:,0] >= 0,
            np.logical_and( new_mask_idx[:,0] < msk.size[2],
            np.logical_and( new_mask_idx[:,1] >= 0,
            np.logical_and( new_mask_idx[:,1] < msk.size[1],
            np.logical_and( new_mask_idx[:,2] >= 0,
                new_mask_idx[:,2] < msk.size[0] )))))
        z, y, x =  np.transpose(new_mask_idx[in_bounds,:])
        expansion_mask.data[z,y,x] = 1
    return expansion_mask

def expand(msk, expansion):
    '''
    Expand the binary mask surface by the given expansion in physical units (e.g., mm, cm, ...)

    Note that all expansions should be positive values!

    Positional arguments:
        :msk:       mask to transform
        :expansion: float or list of floats defining the expansion

    Note: Expansion may be given as:
        - A single number:       uniform expansion in all dimensions
        - A list of 1 number:    uniform expansion in all dimensions
        - A list of 3 numbers:   expansion in x, y, and z, respectively
        - A list of 6 numbers:   expansion in -x, +x, -y, +y, -z, and +z, respectively
    '''
    expansion_mask = expand_surface(msk, expansion)
    expansion_mask.data = np.logical_or(expansion_mask.data, msk.data)
    return expansion_mask

def contract(msk, contraction):
    '''
    Contract the binary mask by the given expansion in physical units (e.g., mm, cm, ...)

    Note that all expansions should be positive values!

    Positional arguments:
        :msk:           mask to transform
        :contraction:   float or list of floats defining the expansion

    Note: Expansion may be given as:
        - A single number:       uniform expansion in all dimensions
        - A list of 1 number:    uniform expansion in all dimensions
        - A list of 3 numbers:   expansion in x, y, and z, respectively
        - A list of 6 numbers:   expansion in -x, +x, -y, +y, -z, and +z, respectively
    '''
    contraction_mask = expand_surface(msk, contraction)
    contraction_mask.data = np.logical_and(msk.data,
        np.logical_not(contraction_mask.data))
    return contraction_mask

def shells(msk, expansions=[], contractions=[]):
    '''
    Create shells from a list of contractions and expansions.

    Positional arguments:
        :msk:   binary mask to contract to create shells
    Keyword arguments:
        :cts:   list of contractions
        :exp:   list of expansions
    Returns:
        - List of expansion and contraction factors, and
        - List of masks representing the shells, where each shell sits inside the previous.
        The index of each mask corresponds to the index of each expansion/contraction factor
    '''
    orig = deepcopy(msk)
    # Order the expansions and contractions
    exp = deepcopy(expansions)
    cts = deepcopy(contractions)
    exp.sort(reverse=True)
    cts.sort(reverse=False)
    # List of expansion and contraction factors defining the shells
    # bounds = deepcopy(exp)
    bounds = ["+"+str(e) for e in exp]
    bounds.extend(["+0.0"])
    bounds.extend(["-"+str(c) for c in cts])

    # Create expanded and contracted masks
    exps = [expand(orig, e) for e in exp]
    cons = [contract(orig, c) for c in cts]
    # Create a list of all expanded and contracted masks
    shls = deepcopy(exps)
    shls.extend([orig])
    shls.extend(cons)

    # Create shells from the expanded and contracted masks
    for idx, s in enumerate(shls):
        if idx < len(shls)-1:
            s.data = np.logical_and(s.data, np.logical_not(shls[idx+1].data))

    return bounds, shls

def centerOfMass(msk):
    '''
    Calculate the center of mass for a mask.

    Positional arguments:
        :msk:   binary mask to calculate for
    Returns:
        Center of mass of the mask in a numpy array.
    '''
    pts = msk.data.nonzero()
    com = np.mean(pts, axis=1)
    return com

def octantsAroundPoint(msk, pt):
    '''
    Create octants around a point in the dose grid. Specify a point to be the center of the octants.

    Positional arguments:
        :msk:   binary mask to contract to create octants
        :pt:    center of octants to cut around
    Returns:
        List of mask objects representing each octant.

    Note:
        Octants are defined as follows:

        \+ indicates the the octant include the positive direction of the corresponding axis.

        +--------+-----------+
        | Index  | Axis      |
        +========+===========+
        |  i     | (x,y,z)   |
        +--------+-----------+
        |  0     | (+,+,+)   |
        +--------+-----------+
        |  1     | (-,+,+)   |
        +--------+-----------+
        |  2     | (-,-,+)   |
        +--------+-----------+
        |  3     | (+,-,+)   |
        +--------+-----------+
        |  4     | (+,+,-)   |
        +--------+-----------+
        |  5     | (-,+,-)   |
        +--------+-----------+
        |  6     | (-,-,-)   |
        +--------+-----------+
        |  7     | (+,-,-)   |
        +--------+-----------+

    '''
    orig = msk.data
    octantMasks = []

    for i in range(8):
        aMask = deepcopy(msk)
        if i != 0:
            aMask.data[pt[0]:, pt[1]:, pt[2]:] = 0
        if i != 1:
            aMask.data[0:pt[0], 0:pt[1], 0:pt[2]] = 0
        if i != 2:
            aMask.data[pt[0]:, 0:pt[1], 0:pt[2]] = 0
        if i != 3:
            aMask.data[0:pt[0], pt[1]:, 0:pt[2]] = 0
        if i != 4:
            aMask.data[0:pt[0], 0:pt[1], pt[2]:] = 0
        if i != 5:
            aMask.data[pt[0]:, pt[1]:, 0:pt[2]] = 0
        if i != 6:
            aMask.data[pt[0]:, 0:pt[1], pt[2]:] = 0
        if i != 7:
            aMask.data[0:pt[0], pt[1]:, pt[2]:] = 0
        octantMasks.append(aMask)

    # Reorder the masks to follow proper labeling
    octs = []
    oct_order = [0,6,2,5,7,1,3,4]
    for i in oct_order:
        octs.append(octantMasks[i])

    return octs

def halves(msk, pt):
    '''
    Cut into superior and inferior halves along the z-axis.

    Positional arguments:
        :msk:   mask object to be cut
        :pt:    point (z,y,x) around which to cut the mask
    Returns:
        List of mask objects representing inferior and superior halves
    '''
    halfMasks = []

    for i in range(2):
        aMask = deepcopy(msk)
        if i == 0:
            aMask.data[0:pt[0], :, :] = 0
        if i == 1:
            aMask.data[pt[0]:, :, :] = 0
        halfMasks.append(aMask)

    return halfMasks

def slices(msk, numSlices, axis):
    '''
    Cut a mask into slices of equal thickness along a specified axis.

    Positional arguments:
        :msk:       mask object to be cut
        :numSlices: number of slices to be created
        :axis:      axis along which to be cut ("x", "y", or "z")
    Returns:
        List of mask objects representing each slice
    '''
    if axis.lower() == "x":
        ax = 0
    elif axis.lower() == "y":
        ax = 1
    elif axis.lower() == "z":
        ax = 2
    else:
        raise ValueError("Axis must be either x, y, or z.")

    bounds = msk.getBounds()
    sliceBounds = []

    # Compute spacing and produce a list of slice bounds
    spacing = (bounds[1][ax]-bounds[0][ax])/numSlices
    b = bounds[0][ax]
    sliceBounds.append(b)
    for i in range(numSlices-1):
        b += spacing
        sliceBounds.append(b)
    sliceBounds.append(-1)

    sliceMasks = []

    for i in range(numSlices):
        # New mask object that will be modified
        slMask = mask()
        slMask.spacing = msk.spacing
        slMask.origin = msk.origin
        slMask.size = msk.size
        slMask.update_end()

        # Data that will be used to comppute slice mask data
        dataA = deepcopy(msk.data)

        if ax == 0:
            dataA[:, :, sliceBounds[i]:sliceBounds[i+1]] = 0
        elif ax == 1:
            dataA[:, sliceBounds[i]:sliceBounds[i+1], :] = 0
        elif ax == 2:
            dataA[sliceBounds[i]:sliceBounds[i+1], :, :] = 0

        slMask.data = np.logical_xor(msk.data, dataA)
        sliceMasks.append(slMask)

    return sliceMasks

def combine_masks(masks, weights=None):
    '''
    Combine multiple masks into one.

    Positional arguments:
        :masks:     list of mask objects to combine
    Returns:
        Mask object that is the combination of all masks given.
    Raises:
        :ValueError:    if only one mask is given
        :ValueError:    if mask specifications (dimension, origin, etc.) do not match
    '''
    # If there is only one mask given, there's nothing to do
    if len(masks) < 2:
        raise ValueError('Only one mask. Nothing to combine.')

    # Check that all specifications match
    specs = {}
    specs['dimension'] = set([m.dimension for m in masks])
    specs['origin'] = set([str(m.origin) for m in masks])
    specs['end'] = set([str(m.end) for m in masks])
    specs['index'] = set([str(m.index) for m in masks])
    specs['size'] = set([str(m.size) for m in masks])
    specs['spacing'] = set([str(m.spacing) for m in masks])
    specs['direction'] = set([str(m.direction) for m in masks])
    for k in specs.keys():
        if len(specs[k]) != 1:
            raise ValueError('Too many different values for field, {}: {}'.format(k, list(specs[k])))


    # If weights for each mask are specified
    if weights:
        if len(weights)!=len(masks):
            raise ValueError('{} masks can not be mapped to {} weights'.format(len(masks), len(weights)))
        # Combine the masks
        comb_mask = deepcopy(masks[0])
        comb_mask.data *= weights[0]
        for i, m in enumerate(masks):
            comb_mask.data += m.data*weights[i]
    # Otherwise, just add the given masks
    else:
        # Combine the masks
        comb_mask = deepcopy(masks[0])
        for m in masks:
            comb_mask.data += m.data

    return comb_mask

def downsample(msk, fracs):
    '''
    Reduce the number of points by uniformly sampling along each axis.

    Positional arguments:
        :msk:   mask to be downsampled
        :fracs: fraction of each axis to be left.
    Returns:
        resampled mask

    Note:
        fracs can be given as either:
            1 value to scale all axes by the same amount
            list of 3 values to scale (x,y,z) axes respectively

        TODO: Does not support interpolation, so it is best that fracs can be expressed as 1/x where x is a whole number
    '''
    # Get fracs in the form of a list with 3 values
    fracs = np.multiply(np.ones(3), fracs)

    # Check that all fracs are < 1
    if not all(b <= 1 for b in fracs):
        raise ValueError("Fraction parameter(s) must be <= 1.")

    # Make a new mask
    new_msk = deepcopy(msk)
    # print "Downsampling {} points".format(len(msk.data.nonzero()[0]))
    # Number of indices to skip in (x,y,z) directions
    skips = [int(1/i) for i in fracs]
    # New data
    new_msk.data = msk.data[::skips[2], ::skips[1], ::skips[0]]
    new_msk.spacing = tuple(np.multiply(msk.spacing, skips))
    new_msk.size = new_msk.data.shape[::-1]
    # print "Reduced to {} points".format(len(new_msk.data.nonzero()[0]))
    return new_msk

def crop(msk):
    '''
    Crop a mask to the bounds of the nonzero values.

    Positional arguments:
        :msk:   mask to crop
    Returns:
        cropped mask
    '''
    new_mask = deepcopy(msk)
    bnds = new_mask.getBounds()
    new_mask.data = new_mask.data[bnds[0][2]:bnds[1][2],bnds[0][1]:bnds[1][1], bnds[0][0]:bnds[1][0]]
    new_mask.origin = new_mask.origin + bnds[0]*new_mask.spacing
    new_mask.end = new_mask.origin + bnds[1]*new_mask.spacing
    new_mask.size = tuple(bnds[1]-bnds[0])
    return new_mask
