import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import color
from skimage import io
import skimage.color
from skimage.color import rgb2gray
import numpy as np

transform_matrix = np.array([[0.299, 0.587, 0.114], [0.59590059, -0.27455667, -0.32134392], [0.21153661, -0.52273617,
                                                                                             0.31119955]])
YIQ_SHAPE=2
GREYSCALE_REPRESENTATION=1
RGB_REPRESENTATION=2
PIXELS=255
def read_image(filename, representation):
    if representation == GREYSCALE_REPRESENTATION:
        img = mpimg.imread(filename)
        img = img.astype('float64')/PIXELS
    else:
        img = io.imread(filename, as_gray=True)
    return img


def imdisplay(filename, representation):
    image = read_image(filename, representation)
    if representation == GREYSCALE_REPRESENTATION:
        plt.imshow(image, cmap='gray')
        plt.show()
        return
    plt.imshow(image)
    plt.show()


def rgb2yiq(imRGB):
    return np.dot(imRGB.reshape(-1, 3), transform_matrix.transpose()).reshape(imRGB.shape)


def yiq2rgb(imYIQ):
    return np.dot(imYIQ.reshape(-1, 3), np.linalg.inv(transform_matrix).transpose()).reshape(imYIQ.shape)


def get_num_pixels(img):
    width, height = img.size
    return width * height


def stretch(ch):
    k = np.nonzero(ch)[0][0]
    x = PIXELS * (ch - ch[k])
    y = (ch[PIXELS] - ch[k])
    return np.round(x/y)


def is_rgb(img):
    return len(img.shape) != YIQ_SHAPE


def histogram_equalize(im_orig):
    lst = []
    rgb_flag = is_rgb(im_orig)
    if rgb_flag:
        original_rgb = rgb2yiq(im_orig)
        im_orig = original_rgb[:, :, 0]
    im_orig = im_orig * PIXELS
    hist_orig, bin_edges = np.histogram(im_orig, bins=PIXELS+1, range=(0, PIXELS))
    chistogram_array = np.cumsum(hist_orig)
    # chistogram_array = 255 * chistogram_array / chistogram_array[-1]
    chistogram_array = stretch(chistogram_array)
    new_image = chistogram_array[(im_orig ).astype(np.uint)] / PIXELS
    if rgb_flag:
        original_rgb[:,:,0] = new_image
        new_image = yiq2rgb(original_rgb)
    hist_eq, bin_eq = np.histogram(im_orig, bins=PIXELS+1, range=(0, PIXELS))
    return [new_image,hist_orig,hist_eq]

def arrange_z(n_quant,hist):
    cum_hist = np.cumsum(hist)
    z = []
    for i in range(n_quant+1):
        raf = (cum_hist[-1] / n_quant)*i
        ind=np.argmax(cum_hist >= raf)
        z.append(ind)
    z[0]=-1
    return z


def quantize(im_orig, n_quant, n_iter):
    rgb_flag=is_rgb(im_orig)
    if rgb_flag:
        original_rgb = rgb2yiq(im_orig)
        im_orig = original_rgb[:, :, 0]
    hist = np.histogram(im_orig , bins=PIXELS+1, range=(0, 1))[0]
    z = arrange_z(n_quant,hist)
    q=np.zeros(n_quant)
    prev_z = z.copy()
    prev_q = q.copy()
    error = []
    for i in range(n_iter):
        for j in range(n_quant):
            all_histograms = hist[round(max(0, int(z[j]))) : round(int(z[j+1]))]
            all_g = np.arange(round(max(0, int(z[j])))+1, round(int(z[j+1]+1)))
            sum_mone = np.dot(all_histograms,all_g)
            sum_mechane = np.sum(all_histograms)
            to_qj = sum_mone/sum_mechane
            q[j] = to_qj
        for j in range(1,n_quant):
            z[j] = (q[j-1] + q[j])/2
        error.append(calc_err(n_quant,q,z,hist))
        if np.array_equal(prev_z, z) & np.array_equal(prev_q, q):
            break
        prev_z=z.copy()
        prev_q=q.copy()
    z = np.divide(z,PIXELS)
    q = np.divide(q,PIXELS)
    for i in range(n_quant):
        im_orig[(im_orig>z[i]) & (im_orig<z[i+1])] = q[i]
    if rgb_flag:
        original_rgb[:, :, 0] = im_orig
        im_orig = yiq2rgb(original_rgb)

    return [im_orig,error]

def calc_err(n_quant,q,z,hist):
    cur_sum = 0
    for i in range(n_quant):
        all_zs=np.arange(max(0,round(z[i]))+1, round(z[i+1]+1))
        lhs = np.subtract(q[i],all_zs)
        lhs=np.power(lhs,2)
        first_slice = max(0,int(round(z[i])))+1
        second_slice = int (round(z[i+1]))
        rhs = hist[first_slice: second_slice+1]
        cur_sum += np.dot(lhs,rhs)
    return cur_sum
