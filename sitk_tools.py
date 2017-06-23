'''
This module contains functions to integrate SimpleITK
into the Oncospace Analytics Framework.
Functions include transformations between data structures and IPython tools.
'''

import SimpleITK as sitk
import matplotlib.pyplot as plt

from framework.Utils.image import mask

def image_from_mask(mask):
    '''
    Convert a mask into a SimpleITK image

    Positional arguments:
        :mask:  mask object
    '''
    myimage = sitk.GetImageFromArray(mask.data.astype(int))
    myimage.SetOrigin(mask.origin)
    myimage.SetSpacing(mask.spacing)
    return myimage

def mask_from_image(img):
    '''
    Convert a SimpleITK image into a mask

    Positional arguments:
        :img:   SimpleITK image object
    '''
    mymask = mask()
    mymask.data = sitk.GetArrayFromImage(img)
    mymask.origin = img.GetOrigin()
    mymask.spacing = img.GetSpacing()
    return mymask

def print_image_details(img):
    '''
    Print the dimension details of an image: size, origin, direction, and spacing

    Positional arguments:
        :img:   an image or list of images
    '''
    if type(img)==list:
        for i, im in enumerate(img):
            print "Image", i
            print "Size:", im.GetSize()
            print "Origin:", im.GetOrigin()
            print "Direction:", im.GetDirection()
            print "Spacing:", im.GetSpacing()
            print "====="
    else:
        print "Size:", img.GetSize()
        print "Origin:", img.GetOrigin()
        print "Direction:", img.GetDirection()
        print "Spacing:", img.GetSpacing()
    pass


# IPython tools ================================================================

from ipywidgets import interact, fixed
from IPython.display import clear_output

# Callback invoked by the interact ipython method for scrolling through the image stacks of
# the two images (moving and fixed).
def display_images(fixed_image_z, moving_image_z, fixed_npa, moving_npa):
    # Create a figure with two subplots and the specified size.
    plt.subplots(1,2,figsize=(10,8))

    # Draw the fixed image in the first subplot.
    plt.subplot(1,2,1)
    plt.imshow(fixed_npa[fixed_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('fixed image')
    plt.axis('off')

    # Draw the moving image in the second subplot.
    plt.subplot(1,2,2)
    plt.imshow(moving_npa[moving_image_z,:,:],cmap=plt.cm.Greys_r);
    plt.title('moving image')
    plt.axis('off')

    plt.show()

# Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
# of an image stack of two images that occupy the same physical space.
def display_images_with_alpha(image_z, alpha, fixed, moving):
    img = (1.0 - alpha)*fixed[:,:,image_z] + alpha*moving[:,:,image_z]
    plt.imshow(sitk.GetArrayFromImage(img),cmap=plt.cm.Greys_r);
    plt.axis('off')
    plt.show()

# Callback invoked when the StartEvent happens, sets up our new data.
def start_plot():
    global metric_values, multires_iterations

    metric_values = []
    multires_iterations = []

# Callback invoked when the EndEvent happens, do cleanup of data and figure.
def end_plot():
    global metric_values, multires_iterations

    del metric_values
    del multires_iterations
    # Close figure, we don't want to get a duplicate of the plot latter on.
    plt.close()

# Callback invoked when the IterationEvent happens, update our data and display new figure.
def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    # Clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # Plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
    plt.xlabel('Iteration Number',fontsize=12)
    plt.ylabel('Metric Value',fontsize=12)
    plt.show()

# Callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the
# metric_values list.
def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))
