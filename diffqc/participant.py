import os
import nibabel as nib
import numpy as np

import matplotlib
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import ImageGrid

import sklearn.cluster
from skimage import feature

from dipy.viz import regtools
from dipy.align.imaffine import AffineMap
from dipy.segment.mask import median_otsu

from diffqc import helper

def samplingScheme(dwi):
    img = nib.load(dwi['file'])
    bval = np.loadtxt(dwi['bval'])
    bvec = np.loadtxt(dwi['bvec'])

    print(img.affine)

    qval = bval*bvec
    iqval = -qval

    fig = plt.figure(figsize=(10,10))

    ax = fig.add_subplot(111, projection='3d')

    norm = matplotlib.colors.Normalize(vmin=np.min(bval), vmax=np.max(bval), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet_r)
    imapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    smp = ax.scatter(qval[0,:], qval[1,:], qval[2,:], c=mapper.to_rgba(bval), marker='o', s=70)
    ismp = ax.scatter(iqval[0,:], iqval[1,:], iqval[2,:], c=imapper.to_rgba(bval), marker='^', s=70)

    lim = np.ceil(np.max(np.abs(qval))/100)*100

    ax.set_xlim3d(-lim, lim)
    ax.set_ylim3d(-lim, lim)
    ax.set_zlim3d(-lim, lim)

    ax.set_aspect('equal', 'box')
    ax.set_title('acquisition scheme ' + dwi['subject_label'])

    plot_name = 'sampling_scheme.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()

def getShells(dwi):
    bval = np.loadtxt(dwi['bval'])
    ub = np.unique(bval)
    k = list(np.isclose(ub[1:],ub[:-1], rtol=0.15)).count(False) + 1
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(bval.reshape(-1,1))
    shells = np.round(kmeans.cluster_centers_.ravel(), decimals=-1)
    _, dirs_per_shell = np.unique(kmeans.labels_, return_counts=1)
    sortind = np.argsort(shells)
    shellind = np.argsort(sortind)
    shells = shells[sortind]
    dirs_per_shell = dirs_per_shell[sortind]
    shellind = shellind[kmeans.labels_]
    shells[shells<50] = 0
    dwi['shells'] = shells
    dwi['dirs_per_shell'] = dirs_per_shell
    dwi['shellind'] = shellind

    print("shells: " + str(shells))
    print("# dirs: " + str(dirs_per_shell))


def denoise(dwi):
    dwi['denoised'] = os.path.join(dwi['data_dir'],
                        os.path.split(dwi['file'])[-1].replace("_dwi.", "_denoised."))
    dwi['noise'] = os.path.join(dwi['data_dir'],
                        os.path.split(dwi['file'])[-1].replace("_dwi.", "_noise."))
    cmd = "dwidenoise %s %s -noise %s -force"%(dwi['file'],
                                               dwi['denoised'],
                                               dwi['noise'])
    # print(cmd)
    helper.run(cmd)

    noise = nib.load(dwi['noise'])
    noiseMap = noise.get_data()
    noiseMap[np.isnan(noiseMap)] = 0

    if dwi['flip_sign'][0] < 0:
        noiseMap = noiseMap[::-1,:,:]

    if dwi['flip_sign'][1] < 0:
        noiseMap = noiseMap[:,::-1,:]

    if dwi['flip_sign'][2] < 0:
        noiseMap = noiseMap[:,:,::-1]

    helper.plotFig(noiseMap, 'Noise Map', dwi['voxSize'])

    plot_name = 'noise_map.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()

def brainMask(dwi):
    img = nib.load(dwi['denoised'])
    raw = img.get_data()
    b0 = raw[:,:,:,dwi['shellind']==0]

    if b0.shape[3] > 0:
        b0 = np.mean(b0, axis=3)

    _, b0_mask = median_otsu(b0,2,1)

    dwi['b0'] = b0
    dwi['mask'] = b0_mask

def dtiFit(dwi):

    # DTI Fit to get residuals
    in_file = dwi['denoised']
    dwi['tensor'] = dwi['denoised'].replace("_denoised", "_tensor")
    dwi['dtiPredict'] = dwi['denoised'].replace("_denoised", "_dtFit")

    cmd = "dwi2tensor %s %s -fslgrad %s %s -predicted_signal %s -force"%(
                                               in_file,
                                               dwi['tensor'],
                                               dwi['bvec'],
                                               dwi['bval'],
                                               dwi['dtiPredict'])

    # print(cmd)
    helper.run(cmd)

def faMap(dwi):

    fa_file = os.path.join(dwi['data_dir'],
                os.path.split(dwi['file'])[-1].replace("_dwi.", "_fa" + "."))
    ev1_file = os.path.join(dwi['data_dir'],
                os.path.split(dwi['file'])[-1].replace("_dwi.", "_ev1" + "."))
    cmd = "tensor2metric %s -fa %s -vector %s -num 1 -force" % (
                                        dwi['tensor'],
                                        fa_file,
                                        ev1_file)

    # print(cmd)
    helper.run(cmd)

    fa = nib.load(fa_file)


    faMap = fa.get_data()
    faMap[np.isnan(faMap)] = 0
    faMap = faMap * dwi['mask']

    ev = nib.load(ev1_file)
    ev = ev.get_data()
    ev[np.isnan(ev)] = 0

    if dwi['flip_sign'][0] < 0:
        ev = ev[::-1,:,:]
        faMap = faMap[::-1,:,:]

    if dwi['flip_sign'][1] < 0:
        ev = ev[:,::-1,:]
        faMap = faMap[:,::-1,:]

    if dwi['flip_sign'][2] < 0:
        ev = ev[:,:,::-1]
        faMap = faMap[:,:,::-1]

    helper.plotFig(faMap, 'fractional anisotropy', dwi['voxSize'])

    plot_name = 'fractional_anisotropy' + '.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()

    #print(flip_sign)

    helper.plotTensor(faMap, ev, 'tensor eigenvector')

    plot_name = 'tensor_eigenvector' + '.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()

def mdsMap(dwi):

    img = nib.load(dwi['denoised'])
    bval = np.loadtxt(dwi['bval'])

    mdsMap = img.get_data()
    mdsMap = np.mean(mdsMap[:,:,:,bval > 50], axis=3)

    mdsMap[np.isnan(mdsMap)] = 0
    mdsMap = mdsMap * dwi['mask']

    if dwi['flip_sign'][0] < 0:
        mdsMap = mdsMap[::-1,:,:]

    if dwi['flip_sign'][1] < 0:
        mdsMap = mdsMap[:,::-1,:]

    if dwi['flip_sign'][2] < 0:
        mdsMap = mdsMap[:,:,::-1]

    helper.plotFig(mdsMap, 'mean diffusion signal', dwi['voxSize'])

    plot_name = 'mean_diffusion_signal' + '.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()


def tensorResiduals(dwi):
    bval = np.loadtxt(dwi['bval'])
    bvec = np.loadtxt(dwi['bvec'])
    img = nib.load(dwi['denoised'])
    raw = img.get_data()
    img_tensor = nib.load(dwi['dtiPredict'])
    tensor_estimator = img_tensor.get_data()
    raw[np.isnan(raw)] = 0
    raw[np.isinf(raw)] = 0
    tensor_estimator[np.isnan(tensor_estimator)] = 0
    tensor_estimator[np.isinf(tensor_estimator)] = 0
    res = np.sqrt((raw.astype(float) - tensor_estimator.astype(float))**2)

    b0 = dwi['b0']

    res[:,:,:,bval<=50] = 0
    res[:,:,:,np.bitwise_and(bval>50, sum(bvec)==0)] = 0

    min_thresh = np.min(b0)
    max_thresh = np.max(b0)
    med_thresh = np.median(b0[b0>0])

    b0_mask = dwi['mask']

    mask = np.repeat(np.expand_dims(np.invert(b0_mask), axis=3), raw.shape[3], axis=3)

    res[mask] = 0

    res[np.isnan(res)] = 0
    res[np.isinf(res)] = 0

    res[res<min_thresh] = 0
    res[res>max_thresh] = 0

    res[mask]=0


    if dwi['flip_sign'][0] < 0:
        raw = raw[::-1,:,:,:]

    if dwi['flip_sign'][1] < 0:
        raw = raw[:,::-1,:,:]

    if dwi['flip_sign'][2] < 0:
        raw = raw[:,:,::-1,:]

    # Plot tensor residuals
    res = helper.normImg(res)

    sl_res = np.sum(np.sum(res, axis=0), axis=0)
    sl_res.shape

    z, diff = np.unravel_index(np.argsort(sl_res, axis=None)[-9:],sl_res.shape)

    fig = plt.figure(figsize=(20,20))
    grid = ImageGrid(fig,111,nrows_ncols=(3,3), axes_pad=0)

    plt.subplots_adjust(wspace=0, hspace=0)
    cnt=0
    for i in range(3):
        for j in range(3):
            pltimg = raw[:,::-1,z[cnt],diff[cnt]].T
            grid[cnt].imshow(pltimg, cm.gray, interpolation='none')
            grid[cnt].axis('off')
            cnt = cnt + 1

    shells = np.unique(dwi['shellind'])
    grid[1].set_title('outlier slices according to tensor residuals', fontsize=16)

    plot_name = 'tensor_residuals' + '.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()

    # Plot Intensity Values per shell
    fig, ax = plt.subplots(nrows=shells.size, ncols=3, figsize=(15,3*shells.size))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    ax[0][0].set_title('transversal')
    ax[0][1].set_title('coronal')
    ax[0][2].set_title('sagittal')

    for i in range(dwi['shells'].size):
        ax[i][0].set(ylabel = 'b = ' + str(int(dwi['shells'][i])))

    for i in range(bval.shape[0]):
        ax[dwi['shellind'][i]][0].plot(np.mean(np.mean(raw[:,:,:,i],axis=0),axis=0))
        ax[dwi['shellind'][i]][1].plot(np.mean(np.mean(raw[:,:,:,i],axis=2),axis=0))
        ax[dwi['shellind'][i]][2].plot(np.mean(np.mean(raw[:,:,:,i],axis=1),axis=1))

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i][j].axis('on')

    plot_name = 'intensity_values' + '.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()

def anatOverlay(dwi,t1):
    if t1['file'].split("acq-")[-1] != t1['file']:
        t1_acq = '_acq-' + t1['file'].split("acq-")[-1].split("_")[0]
    else:
        t1_acq = ''

    imgT1 = nib.load(t1['file'])
    img = nib.load(dwi['denoised'])

    b0_affine = img.affine
    b0 = dwi['b0']

    b0_mask = dwi['mask']

    b0 = b0 * b0_mask
    t1 = imgT1.get_data()
    t1_affine = imgT1.affine

    (t1_affine, perm, flip_sign) = helper.fixImageHeader(imgT1)

    t1 = np.transpose(t1, perm)

    if flip_sign[0] < 0:
        t1 = t1[::-1,:,:]

    if flip_sign[1] < 0:
        t1 = t1[:,::-1,:]

    if flip_sign[2] < 0:
        t1 = t1[:,:,::-1]

    affine_map = AffineMap(np.eye(4),
                           t1.shape, t1_affine,
                           b0.shape, b0_affine)

    resampled = affine_map.transform(np.array(b0))

    # Normalize the input images to [0,255]
    t1 = helper.normImg(t1)
    b0 = helper.normImg(resampled)

    overlay = np.zeros(shape=(t1.shape) + (3,), dtype=np.uint8)
    b0_canny = np.zeros(shape=(t1.shape), dtype=np.bool)

    ind = helper.getImgThirds(t1)

    for i in ind[0]:
        b0_canny[:,:,i] = feature.canny(b0[:,:,i], sigma=1.5)

    for i in ind[1]:
        b0_canny[:,i,:] = feature.canny(np.squeeze(b0[:,i,:]), sigma=1.5)

    for i in ind[2]:
        #b0_canny[i-1,:,:] = feature.canny(np.squeeze(b0[i-1,:,:]), sigma=1.5)
        b0_canny[i,:,:] = feature.canny(np.squeeze(b0[i,:,:]), sigma=1.5)
        #b0_canny[i+1,:,:] = feature.canny(np.squeeze(b0[i+1,:,:]), sigma=1.5)

    overlay[..., 0] = t1
    overlay[..., 1] = t1
    overlay[..., 2] = t1
    overlay[..., 0] = b0_canny*255

    voxSize = imgT1.header['pixdim'][1:4]

    helper.plotFig(overlay, 'alignment DWI -> T1', voxSize) #[perm])
    plot_name = 't1' + t1_acq + '_overlay.png'
    plt.savefig(os.path.join(dwi['fig_dir'], plot_name), bbox_inches='tight')
    plt.close()
