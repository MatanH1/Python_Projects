import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy
import numpy as np
import skimage
from skimage import io
import os
from scipy.linalg import pascal
from scipy.ndimage import convolve1d
from scipy.signal import convolve2d


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def build_gaussian_pyramid(im, max_levels, filter_size):
    pyr = [im]
    cur_img = im
    filter_vec = build_filter(filter_size)
    for i in range(max_levels):
        cur_img = convolve2d(cur_img, filter_vec,mode='same')
        cur_img = convolve2d(cur_img, filter_vec.T,mode='same')
        cur_img = reduce_img(cur_img)
        pyr.append(cur_img)
    return [pyr, filter_vec]

def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr = []
    gaussian = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    cur_img = im
    next_img = reduce_img(im)
    filter_vec = build_filter(filter_size)*2
    for i in range(max_levels-1):
        expanded = gaussian[i+1]
        while np.shape(expanded) != np.shape(gaussian[i]):
            expanded = expand_img(expanded)
            expanded = convolve2d(expanded, filter_vec, mode='same')
            expanded = convolve2d(expanded, filter_vec.T, mode='same')
        to_add = gaussian[i] - expanded
        pyr.append(to_add)
    pyr.append(gaussian[max_levels-1])
    return [pyr,filter_vec/2]


def reduce_img(im):
    im = np.delete(im, list(range(1, np.shape(im)[0], 2)), axis=0)
    im = np.delete(im, list(range(1, np.shape(im)[1], 2)), axis=1)
    return im


def expand_img(im):
    expanded = np.zeros(2*np.array(np.shape(im)))
    expanded[::2,::2] = im
    return expanded


def build_filter(filter_size):
    gaus_filter = pascal(filter_size, 'lower')[-1]
    cumsum = np.cumsum(gaus_filter)[-1]
    gaus_filter = np.divide(gaus_filter, cumsum)
    return gaus_filter.reshape(1, filter_size)

def laplacian_to_image(lpyr, filter_vec, coeff):
    lpyr = multiply_coef(coeff,lpyr)
    cur_img = lpyr[-1]
    for i in range(len(lpyr)-1,0,-1):
        new_img = expand_img(cur_img)
        new_img = scipy.ndimage.filters.convolve(new_img, filter_vec * 2)
        new_img = scipy.ndimage.filters.convolve(new_img, (filter_vec * 2).T)
        cur_img = new_img + lpyr[i-1]
    return cur_img

def multiply_coef(coeff,lpyr):
    for i in range(len(lpyr)):
        lpyr[i] = np.multiply(lpyr[i], coeff[i])
    return lpyr
def render_pyramid(pyr, levels):
    pyr=pyr[0]
    for i in pyr:
        i = np.divide(i,np.amax(i))
    images = pyr[0]
    for i in range(1,levels):
        shape_to_fill_cols = (np.power(2,i)-1)*np.shape(pyr[i])[1]
        shape_to_fill_rows = np.shape(pyr[i])[0]
        shape_to_fill = np.zeros((shape_to_fill_cols,shape_to_fill_rows))
        new_im = np.concatenate((pyr[i],shape_to_fill),axis=0)
        images = np.concatenate((images,new_im),axis=1)
    return images



def display_pyramid(pyr,levels):
    image = render_pyramid(pyr,levels)
    plt.imshow(image, cmap='gray')
    plt.show()


def read_image(filename,representation):
    if representation == 2:
        img = mpimg.imread(filename)
        if (img[0][0][0]>1):
            img = img.astype('float64') /256
    else:
        img = io.imread(filename, as_gray=True)
    return img

def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    l1 = build_laplacian_pyramid(im1,max_levels,filter_size_im)[0]
    l2 = build_laplacian_pyramid(im2,max_levels,filter_size_im)[0]
    #mask = mask_to_binary(mask)
    gm = build_gaussian_pyramid(mask,max_levels,filter_size_mask)
    lout = []
    for i in range(max_levels):
        lout.append(np.multiply(gm[0][i],l1[i])+np.multiply((1-gm[0][i]),l2[i]))
    return laplacian_to_image(lout,gm[1],np.ones(len(lout)))

def mask_to_binary(mask):
    mask[mask > 0] = True
    mask[mask == 0] = False

    return mask


def blending_example1():
    im1 = read_image(relpath('brock.jpg'), 2)
    im2 = read_image(relpath('obama.png'), 2)
    mask = mask_to_binary(read_image(relpath('new_mask.png'),1))
    blended = np.empty(np.shape(im1))
    for i in range(3):
        blended[:,:,i] = pyramid_blending(im1[:,:,i],im2[:,:,i],mask,8,3,5)
    fig, axs= plt.subplots(2,2)
    axs[0,0].imshow(im1)
    axs[0,1].imshow(im2)
    axs[1,0].imshow(mask,cmap=plt.get_cmap('gray'))
    axs[1,1].imshow(blended)
    plt.show()

    return im1, im2, mask, blended

def blending_example2():
        im2 = read_image(relpath('pika.png'), 2)
        im1 = read_image(relpath('shrek.jpg'), 2)
        mask = read_image(relpath('shrek_mask.jpg'), 1)
        blended = np.empty(np.shape(im1))
        for i in range(3):
            blended[:, :, i] = pyramid_blending(im1[:, :, i], im2[:, :, i], mask, 8, 3, 5)
        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(im1)
        axs[0, 1].imshow(im2)
        axs[1, 0].imshow(mask, cmap=plt.get_cmap('gray'))
        axs[1, 1].imshow(blended)
        plt.show()

        return im1, im2, mask, blended

#a = np.array([[1, 2], [3, 4]])
#b = np.array([[5],[6]]).T
#print(np.concatenate((a,b),axis=1))

#blending_example1()
blending_example2()
#im1 = read_image('brock_mask.png')
#im2 = read_image('obama.png')
#display_pyramid(pyr,4)