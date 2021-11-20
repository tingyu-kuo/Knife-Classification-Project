import numpy as np
from PIL import Image
from pathlib import Path
import random
from matplotlib import pyplot as plt
from load_images import plot_3d, plot_2d


class Random_Sample():
    def __init__(self):
        pass

    def random_maxpool(self, array=None, kernal_size=10, stride=10, top=5):
        a, k, s = np.array(array), kernal_size, stride
        h, w = a.shape
        self.top = top
        new_array = np.array([])
        new_size = [int((h-k)/s)+1, int((h-k)/s)+1]

        if s > k:
            raise BaseException('expected " kernal_size > stride " but got " kernal_size < stride " instead.')
        if (h-k)%s != 0:
            raise BaseException('please check your kernal_size and stride')

        # sliding window
        for i in range(0, h, s):
            for j in range(0, w, s):
                sub_a = a[i:i+k, j:j+k]
                new_value = self.random_sample(sub_a)
                new_array = np.append(new_array, new_value)
        new_array = np.reshape(new_array, new_size)
        return new_array.astype(int)

    def random_sample(self, array):
        tmp = array.copy()
        top_num = random.randrange(self.top)
        for _ in range(top_num-1):
            # set top1 to 0 and pick top2
            max_pos = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)
            tmp[max_pos] = 0
        max_value = np.max(tmp)
        return int(max_value)

def plot_3d(img):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # X, Y, Z information
    y, x = img.shape
    X = np.arange(0, x)
    Y = np.arange(0, y)
    X, Y= np.meshgrid(X, Y)
    Z = img
    ax.set_zlim(0, 50)
    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        linewidth=0,
        antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()
    # reference: https://matplotlib.org/stable/gallery/mplot3d/surface3d.html


if __name__ == '__main__':
    rs = Random_Sample()
    data = Image.open(Path('data', 'Batch1', 'B', '1', 'LWearDepthRaw.Tif'))
    plot_2d(np.array(data))
    new = rs.random_maxpool(data, kernal_size=10, stride=10, top=5)
    # plot_3d(new)
    plot_2d(new)

