import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import skimage
import imageio
from scipy.io import wavfile
from skimage import io
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
from scipy.signal import istft, stft, convolve2d

conv_ker_x = np.array([[0.5],[0],[-0.5]])
def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


def create_dft_matrix(n,arg):
    """

    :param n: len of signal
    :param arg: -1 if coming from
    :return: matrix that by multiplying it we get the DFT
    """
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.multiply(i,j)
    omega = np.exp(arg* 2 * np.pi *a* 1j / n)
    return omega
def DFT(signal):
    """

    :param signal:  array to find DFT of
    :return: DFT array
    """
    dim = np.shape(signal)
    signal = np.squeeze(signal)
    mat = create_dft_matrix(len(signal), -1)
    return np.dot(mat,signal).reshape(dim)

def DFT_for_2(signal):
    mat = create_dft_matrix(len(signal), -1)
    return np.dot(mat, signal)
def IDFT(fourier_signal):
    """

    :param fourier_signal: DFT signal to find its origin array
    :return: Inverse DFT
    """
    dim = np.shape(fourier_signal)
    fourier_signal = np.squeeze(fourier_signal)
    mat = create_dft_matrix(len(fourier_signal),1)/len(fourier_signal)
    return np.dot(mat,fourier_signal).reshape(dim)

def IDFT_for_2(fourier_signal):
    mat = create_dft_matrix(len(fourier_signal), 1) / len(fourier_signal)
    return np.dot(mat, fourier_signal)

def DFT2(image):
    """

    :param image: 2D image
    :return: 2D Fouier Trasnform signal
    """
    dim = np.shape(image)
    new_img = np.zeros(dim).astype(np.complex128)
    for i in range(dim[0]):
        new_img[i,:]=DFT(image[i,:])
    for i in range(np.shape(new_img)[1]):
        new_img[:,i]=DFT(new_img[:,i])
    return np.asarray(new_img).reshape(dim)

def IDFT2(fourier_image):
    """

    :param fourier_image: 2D Image after Fourier transform
    :return: Inverse 2D Fouier Trasnform signal
    """
    dim = np.shape(fourier_image)
    new_img = np.zeros(dim).astype(np.complex128)
    for i in range(dim[0]):
        new_img[i, :] = IDFT(fourier_image[i, :])
    for i in range(np.shape(new_img)[1]):
        new_img[:, i] = IDFT(new_img[:, i])
    return np.asarray(new_img).reshape(dim)


def change_rate(filename,ratio):
    """

    :param filename: name of the wav file we will load
    :param ratio: tempo to multiply by
    :return:
    """
    cur_rate, cur_audio = wavfile.read(filename)
    wavfile.write('change_rate.wav',np.round(cur_rate*ratio).astype("int16"), np.round(cur_audio).astype("int16"))

def change_samples(filename,ratio):
    """

    :param filename: name of the wav file we will load
    :param ratio: tempo to multiply by
    :return: new samples the audio will take
    """
    cur_rate, cur_audio = wavfile.read(filename)
    dft_data = DFT(cur_audio)
    centered = np.fft.fftshift(dft_data)
    resized = resize(centered, ratio)
    cur_audio = IDFT(resized)
    shifted = np.fft.ifftshift(cur_audio)
    wavfile.write("change_sample.wav",cur_rate,resized)
    return shifted.astype("float64")


def resize(data,ratio):
    """
    :param data: array to change
    :param ratio: size to multiply by
    :return: new sized array
    """
    new_size = len(data) // ratio
    if new_size>len(data):
        resized = expand(data,new_size)
    else:
        resized = shrink(data,new_size)
    return resized.astype(data.dtype)

def expand(data,new_size):
    """

    :param data: array to change
    :param new_size: new size requested
    :return: new sized array
    """
    to_add = new_size - len(data)
    to_add_side = to_add//2
    for i in range(int(to_add_side)):
        data = np.insert(data,0,0)
        data = np.insert(data,len(data)-1,0)
    if to_add%2 != 0:
        data = np.insert(data,0,0)
    return data


def shrink(data,new_size):
    """

    param data: array to change
    :param new_size: new size requested
    :return: new sized array
    """
    to_remove = len(data) - new_size
    for i in range(int(to_remove/2)):
        data=np.delete(data,i)
    for i in range(int(to_remove/2)):
        data =np.delete(data,len(data)-1)
    if to_remove%2 != 0:
        data = np.delete(data,0)
    return data

def resize_spectrogram(data, ratio):
    new_data = stft(data)
    return istft(np.apply_along_axis(resize, 1, new_data, ratio))

def resize_vocoder(data,ratio):
    new_data = stft(data)
    return istft(phase_vocoder(new_data, ratio))

def magnitude(x,y):
    """

    :param x: derivative in x axis
    :param y: derivative in y avis
    :return: magnitude of them calculated
d    """
    return np.sqrt(np.abs(x)**2 + np.abs(y)**2)

def conv_der(im):
    """

    :param im: 2D image
    :return: magnitude of x-axis and y-axis derivative calculated
    """
    y = convolve2d(im,conv_ker_x.T,mode='same')
    x = convolve2d(im,conv_ker_x,mode='same')
    return magnitude(x,y)

def fourier_der(im):
    """

        :param im: 2D image
        :return: magnitude of x-axis and y-axis derivative calculated
        """
    dft = DFT2(im)
    dft_for_x = np.fft.fftshift(dft)
    y, x= np.meshgrid(np.arange(-len(dft)//2,len(dft)//2), np.arange(-len(dft[0])//2,len(dft[0])//2))
    x = np.multiply(dft_for_x,np.transpose(x))
    y = np.multiply(dft_for_x,np.transpose(y))
    x = np.fft.ifftshift(x)
    y = np.fft.ifftshift(y)
    return magnitude(IDFT2(x),IDFT2(y))

def read_image(filename, representation):
    if representation == 2:
        img = mpimg.imread(filename)
        img = img.astype('float64')/256
    else:
        img = io.imread(filename, as_gray=True)
    return img
