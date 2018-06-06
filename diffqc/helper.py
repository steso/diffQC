import os.path
import numpy as np
import nibabel as nib
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid

def getImgThirds(img):
    indx = np.floor(np.linspace(img.shape[2]/3, img.shape[2]-img.shape[2]/3,3)).astype(int)
    indy = np.floor(np.linspace(img.shape[1]/3, img.shape[1]-img.shape[1]/3,3)).astype(int)
    indz = np.floor(np.linspace(img.shape[0]/3, img.shape[0]-img.shape[0]/3,3)).astype(int)
    return [indx, indy, indz]


def normImg(img):
    return 255 * ((img - img.min()) / (img.max() - img.min()))

def fourierSharpness(img):
    f = np.fft.fftn(img, axes=(0, 1, 2))
    AF = abs(np.roll(f, np.array(f.shape)/2, axis=(0, 1, 2)));
    return float(np.count_nonzero(AF > (np.max(AF)/1000))) / float(np.prod(img.shape))

def plotFig(img, title):

    ind=getImgThirds(img)

    fig = plt.figure(figsize=(20,20))
    grid = ImageGrid(fig,111,nrows_ncols=(3,3), axes_pad=0)
    fig.subplots_adjust(wspace=0, hspace=0)

    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=3)
        img = 255 * ((img - img.min()) / (img.max() - img.min()))

    if img.shape[0]<img.shape[1]:
        lr_pad = int((img.shape[1]-img.shape[0]) / 2 + 1)
        img = np.pad(img,[(lr_pad,lr_pad), (0, 0), (0,0), (0,0)],'constant', constant_values=(0, 0))
        ind[2] = ind[2] + lr_pad

    ax = (1, 0, 2)
    cnt=0
    for i in range(3):
        for j in range(3):
            if i==0:
                pltimg = img[:,::-1,ind[i][j],:]

            elif i==1:
                pltimg = img[:,ind[i][j],::-1,:]
            elif i==2:
                pltimg = img[ind[i][j],::-1,::-1,:]

            pltimg = np.transpose(pltimg, axes=ax)

            if len(np.squeeze(pltimg).shape) == 2:
                grid[cnt].imshow(np.squeeze(pltimg), cmap='gray', vmin = 0, vmax = 255, interpolation='none')
            else: #colored
                grid[cnt].imshow(np.squeeze(pltimg), interpolation='none')

            grid[cnt].axis('off')

            cnt = cnt + 1


    grid[0].set(ylabel='transversal')
    grid[0].axis('on')
    grid[0].xaxis.set_visible(False)
    grid[0].yaxis.set_ticks([])
    grid[0].yaxis.label.set_fontsize(16)

    grid[3].set(ylabel='coronal')
    grid[3].axis('on')
    grid[3].xaxis.set_visible(False)
    grid[3].yaxis.set_ticks([])
    grid[3].yaxis.label.set_fontsize(16)

    grid[6].set(ylabel='sagittal')
    grid[6].axis('on')
    grid[6].xaxis.set_visible(False)
    grid[6].yaxis.set_ticks([])
    grid[6].yaxis.label.set_fontsize(16)

    grid[1].set_title(title, fontsize=16)

def fixImageHeader(img):
    # flip dimensions to clean up Header-Trafo
    dims = img.header.get_data_shape();
    dims[:3]
    M = img.affine

    perm = np.argsort(np.square(np.transpose(M[:3,:3])).dot(np.transpose([1, 2, 3])))

    M = M[:,np.insert(perm,3,3)]
    flip_sign = np.sign(M[:3,:3].dot([1, 2, 3]))


    R = M[:3,:3]
    T = M[:3,3]
    orig = np.linalg.inv(R).dot(-T) + 1;

    if flip_sign[0] < 0:
        orig[0] = dims[0] - orig[0] + 1
        M[:,0] = -1 * M[:,0]
        M[:3,3] = -M[:3,:3].dot(orig[:3] - 1)
    if flip_sign[1] < 0:
        orig[1] = dims[1] - orig[1] + 1
        M[:,1] = -1 * M[:,1]
        M[:3,3] = -M[:3,:3].dot(orig[:3] - 1)
    if flip_sign[2] < 0:
        orig[2] = dims[2] - orig[2] + 1
        M[:,2] = -1 * M[:,2]
        M[:3,3] = -M[:3,:3].dot(orig[:3] - 1)

    raw = img.get_data()
    raw = np.transpose(raw, perm)

    if flip_sign[0] < 0:
        raw = raw[::-1,:,:]

    if flip_sign[1] < 0:
        raw = raw[:,::-1,:]

    if flip_sign[2] < 0:
        raw = raw[:,:,::-1]

    return (raw, M)
