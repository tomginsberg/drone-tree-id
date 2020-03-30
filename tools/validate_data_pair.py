from glob import glob
from os.path import join, basename

import fire
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread


def validate(path, plot=False):
    chm = glob(join(path, '*_CHM.tif'))
    assert len(chm) == 1, f'0 or > 1 files matching *_CHM.tif: {chm}'

    ortho = glob(join(path, '*_ortho.tif'))
    assert len(ortho) == 1, f'0 or > 1 files matching *_ortho.tif: {ortho}'

    chm, ortho = imread(chm[0]), imread(ortho[0])
    print(f'Ortho {ortho.shape}, CHM {chm.shape}')
    assert chm.shape[:2] == ortho.shape[:2], f'Print shape mismatch. CHM {chm.shape[:2]}, ortho: {ortho.shape[:2]}'
    assert ortho.shape[-1] == 4, 'Ortho expected to have an alpha channel'
    assert len(chm.shape) == 2, 'CHM expected to be grayscale'

    print(f'Data pair at {path} is valid')
    if plot:
        fig, axs = plt.subplots(2, 2)
        (ax1, ax2), (ax3, ax4) = axs
        fig.suptitle(basename(path).replace('_', ' '))
        ax1.imshow(chm)
        ax2.imshow(ortho)
        chm = np.where(chm == -np.inf, 0, chm)
        chm = (255 * (chm - np.min(chm)) / np.max(chm))
        ortho[:, :, 3] = chm
        ax3.imshow(ortho)
        x, y = np.meshgrid(np.arange(chm.shape[1]), np.arange(chm.shape[0]))
        ax4.contour(x, y, chm)

        for ax in axs.flat:
            ax.label_outer()
            ax.axis('off')
        plt.show()


if __name__ == '__main__':
    fire.Fire(validate)
