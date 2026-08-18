import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for standalone scripts
from skimage.io import imread, imshow, imsave

from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift

def fourier2(im):
    return fftshift(fft2(im))

def ifourier2(f):
    return ifft2(ifftshift(f)).real

def fourier(s):
    return fftshift(fft(s))

def ifourier(f):
    return ifft(ifftshift(f)).real

def ampl(f):
    return np.sqrt(f.real**2 + f.imag**2)

def phase(f):
    return np.arctan2(f.imag, f.real)

## Fig 1: Generate a simple 1D signal
# t = np.arange(1000)
# s = np.sin(t)  # Making the frequency lower for better visualization
# f = fourier(s)

# # Create and display plot
# plt.figure(figsize=(10, 8))
# plt.subplot(2, 1, 1)
# plt.plot(ampl(f))
# plt.title('Amplitude of Fourier Transform')
# plt.xlabel('Frequency')
# plt.ylabel('Amplitude')

# plt.subplot(2, 1, 2)
# plt.plot(s)
# plt.title('Original Signal')
# plt.xlabel('Time')
# plt.ylabel('Amplitude')

# plt.tight_layout()
# plt.show()

## Fig 2: Inverse Fourier Transform
# f = np.zeros((1000,))
# f[500] = 500.
# f[520] = 500.
# f[480] = 400.
# # f[400] = 250.
# # f[600] = 250.
# s = ifourier(f)

# plt.figure()
# plt.subplot(2,1,1)
# plt.plot(ampl(f))
# plt.subplot(2,1,2)
# plt.plot(s)
# plt.show()

## Fig 3: 2D Fourier Transform

# fake_im = np.zeros((500,500)).astype('uint8')
# for i in range(fake_im.shape[1]):
#     fake_im[:,i] = np.sin(i/10.)*127+127
        
# f = fourier2(fake_im)

# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(fake_im, cmap=plt.cm.gray, vmin=0, vmax=255, interpolation='none')
# plt.title('Original')
# plt.axis('off')
# plt.subplot(1,2,2)
# plt.imshow(ampl(f), interpolation='none', cmap='viridis')
# plt.title('Fourier amplitude')
# plt.axis('off')
# plt.show()
# plt.figure()
# plt.plot(ampl(f)[250,:])
# plt.show()

## Fig 4: 2D Inverse Fourier Transform

fake_fourier = np.zeros((500,500)) + 0j # cast to complex by adding a complex number
# fake_fourier[250,255] = 1j
fake_fourier[200,250] = 20.
# fake_fourier[260,260] = 3.
#fake_fourier[252,249] = 6.

im = ifourier2(fake_fourier).real

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(im, cmap=plt.cm.gray, interpolation='none')
plt.title('Reconstructed image')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(ampl(fake_fourier), interpolation='none', cmap='viridis')
# Add horizontal and vertical lines through the center
center_y, center_x = fake_fourier.shape[0] // 2, fake_fourier.shape[1] // 2
plt.axhline(y=center_y, color='r', linestyle='-', alpha=0.5)
plt.axvline(x=center_x, color='r', linestyle='-', alpha=0.5)
plt.title('Fourier amplitude')
plt.axis('off')
plt.show()



