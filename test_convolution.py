import imageio
import matplotlib.pyplot as plt
import numpy as np
import scipy

im = imageio.imread("../marmottons.jpg")
a = np.zeros((im.shape[0], im.shape[1], 3))
a[:, :, 0] = scipy.signal.convolve2d(im[1:-1, 1:-1, 0], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
a[:, :, 1] = scipy.signal.convolve2d(im[1:-1, 1:-1, 1], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
a[:, :, 2] = scipy.signal.convolve2d(im[1:-1, 1:-1, 2], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

plt.imshow(a)
plt.show()