#!/usr/bin/env python3
import argparse
import os
import subprocess
import nibabel as nib
import numpy as np
from glob import glob
from mrtrix3 import app, file, fsl, image, path, run

import matplotlib
matplotlib.use('Agg')
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
__version__ = open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                'version')).read()

def run(command, env={}):
    merged_env = os.environ
    merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = str(line, 'utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d"%process.returncode)

parser = argparse.ArgumentParser(description='Example BIDS App entrypoint script.')
parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir.',
                    choices=['participant', 'group'])
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-"). If this parameter is not '
                   'provided all subjects should be analyzed. Multiple '
                   'participants can be specified with a space separated list.',
                   nargs="+")
parser.add_argument('--skip_bids_validator', help='Whether or not to perform BIDS dataset validation',
                   action='store_true')
parser.add_argument('-v', '--version', action='version',
                    version='BIDS-App example version {}'.format(__version__))


args = parser.parse_args()

if not args.skip_bids_validator:
    run('bids-validator %s'%args.bids_dir)

subjects_to_analyze = []
# only for a subset of subjects
if args.participant_label:
    subjects_to_analyze = args.participant_label
# for all subjects
else:
    subject_dirs = glob(os.path.join(args.bids_dir, "sub-*"))
    subjects_to_analyze = [subject_dir.split("-")[-1] for subject_dir in subject_dirs]

# running participant level
if args.analysis_level == "participant":

    # find all T1s and skullstrip them
    # for subject_label in subjects_to_analyze:
    #     for T1_file in glob(os.path.join(args.bids_dir, "sub-%s"%subject_label,
    #                                      "anat", "*_T1w.nii*")) + glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-*","anat", "*_T1w.nii*")):
    #         out_file = os.path.split(T1_file)[-1].replace("_T1w.", "_brain.")
    #         cmd = "bet %s %s"%(T1_file, os.path.join(args.output_dir, out_file))
    #         print(cmd)
    #         run(cmd)

    # find all DWI files and run denoising and tensor / residual calculation
    for subject_label in subjects_to_analyze:
        # create subj dir
        subject_dir = os.path.join(args.output_dir, 'qc_data', 'sub-' + subject_label)
        fig_dir = os.path.join(args.output_dir, 'qc_figures', 'sub-' + subject_label)
        if not os.path.isdir(subject_dir):
            os.makedirs(subject_dir)
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)

        for dwi_file in glob(os.path.join(args.bids_dir, "sub-%s"%subject_label,
                                          "dwi", "*_dwi.nii*")) + glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-*","dwi", "*_dwi.nii*")):

            # Get DWI sampling scheme
            img = nib.load(dwi_file)
            bval = np.loadtxt(dwi_file.replace("_dwi.nii.gz", "_dwi.bval"))
            bvec = np.loadtxt(dwi_file.replace("_dwi.nii.gz", "_dwi.bvec"))

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
            ax.set_title('acquisition scheme ' + subject_label)

            plot_name = 'sampling_scheme.png'
            plt.savefig(os.path.join(args.output_dir, fig_dir, plot_name), bbox_inches='tight')
            plt.close()

            # get nr of shells and directions
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

            print(shells)
            print(dirs_per_shell)

            dataset = {}
            dataset['shells'] = shells
            dataset['dirs_per_shell'] = dirs_per_shell
            dataset['resolution'] = img.header['pixdim'][1:4]

            # Denoising to obtain noise-map
            out_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_denoised.")
            noise_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_noise.")
            cmd = "dwidenoise %s %s -noise %s -force"%(dwi_file,
                                                       os.path.join(args.output_dir, subject_dir, out_file),
                                                       os.path.join(args.output_dir, subject_dir, noise_file))
            print(cmd)
            run(cmd)

            noise = nib.load(os.path.join(args.output_dir, subject_dir, noise_file))
            noiseMap = noise.get_data()
            noiseMap[np.isnan(noiseMap)] = 0
            helper.plotFig(noiseMap, 'Noise Map')

            plot_name = 'noise_map.png'
            plt.savefig(os.path.join(args.output_dir, fig_dir, plot_name), bbox_inches='tight')
            plt.close()

            img = nib.load(os.path.join(args.output_dir, subject_dir, out_file))

            # # Step 2: Gibbs ringing removal (if available)
            # if unring_cmd:
            #     run.command(unring_cmd + ' dwi_denoised.nii dwi_unring' + fsl_suffix + ' -n 100')
            #     file.delTemporary('dwi_denoised.nii')
            #     unring_output_path = fsl.findImage('dwi_unring')
            #     run.command('mrconvert ' + unring_output_path + ' dwi_unring.mif -json_import input.json')
            #     file.delTemporary(unring_output_path)
            #     file.delTemporary('input.json')




            # DTI Fit to get residuals
            in_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_denoised.")
            out_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_tensor.")
            bvecs_file = dwi_file.replace("_dwi.nii.gz", "_dwi.bvec")
            bvals_file = dwi_file.replace("_dwi.nii.gz", "_dwi.bval")
            fit_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_dtFit.")

            cmd = "dwi2tensor %s %s -fslgrad %s %s -predicted_signal %s -force"%(
                                                       os.path.join(args.output_dir, subject_dir, in_file),
                                                       os.path.join(args.output_dir, subject_dir, out_file),
                                                       bvecs_file,
                                                       bvals_file,
                                                       os.path.join(args.output_dir, subject_dir, fit_file))
            run(cmd)

            # Create FA maps
            in_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_tensor.")
            out_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_fa.")
            cmd = "tensor2metric %s -fa %s -force" % (
                                                os.path.join(args.output_dir, subject_dir, in_file),
                                                os.path.join(args.output_dir, subject_dir, out_file))
            run(cmd)

            fa = nib.load(os.path.join(args.output_dir, subject_dir, out_file))

            raw = img.get_data()
            b0 = raw[:,:,:,shellind==0]

            if b0.shape[3] > 0:
                b0 = np.mean(b0, axis=3)

            _, b0_mask = median_otsu(b0,2,1)

            faMap = fa.get_data()
            faMap[np.isnan(faMap)] = 0
            faMap = faMap * b0_mask

            helper.plotFig(faMap, 'Fractional Anisotropy')

            plot_name = 'fractional_anisotropy.png'
            plt.savefig(os.path.join(args.output_dir, fig_dir, plot_name), bbox_inches='tight')
            plt.close()

            # Calc DTI residuals
            raw = img.get_data()
            img_tensor = nib.load(os.path.join(args.output_dir, subject_dir, fit_file))
            tensor_estimator = img_tensor.get_data()
            raw[np.isnan(raw)] = 0
            raw[np.isinf(raw)] = 0
            tensor_estimator[np.isnan(tensor_estimator)] = 0
            tensor_estimator[np.isinf(tensor_estimator)] = 0
            res = np.sqrt((raw.astype(float) - tensor_estimator.astype(float))**2)
            b0 = raw[:,:,:,shellind==0]
            if b0.shape[3] > 0:
                b0 = np.mean(b0, axis=3)

            res[:,:,:,bval<=50] = 0
            res[:,:,:,np.bitwise_and(bval>50, sum(bvec)==0)] = 0

            min_thresh = np.min(b0)
            max_thresh = np.max(b0)
            med_thresh = np.median(b0[b0>0])

            _, b0_mask = median_otsu(b0,2,1)

            mask = np.repeat(np.expand_dims(np.invert(b0_mask), axis=3), raw.shape[3], axis=3)

            res[mask] = 0

            res[np.isnan(res)] = 0
            res[np.isinf(res)] = 0

            res[res<min_thresh] = 0
            res[res>max_thresh] = 0

            res[mask]=0

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

            grid[1].set_title('outlier slices according to tensor residuals', fontsize=16)

            plot_name = 'tensor_residuals.png'
            plt.savefig(os.path.join(args.output_dir, fig_dir, plot_name), bbox_inches='tight')
            plt.close()

            # Plot Intensity Values per shell
            fig, ax = plt.subplots(nrows=shells.size, ncols=3, figsize=(15,3*shells.size))
            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            ax[0][0].set_title('transversal')
            ax[0][1].set_title('coronal')
            ax[0][2].set_title('sagittal')

            for i in range(shells.size):
                ax[i][0].set(ylabel = 'b = ' + str(int(shells[i])))

            for i in range(bval.shape[0]):
                ax[shellind[i]][0].plot(np.mean(np.mean(raw[:,:,:,i],axis=0),axis=0))
                ax[shellind[i]][1].plot(np.mean(np.mean(raw[:,:,:,i],axis=2),axis=0))
                ax[shellind[i]][2].plot(np.mean(np.mean(raw[:,:,:,i],axis=1),axis=1))

            for i in range(ax.shape[0]):
                for j in range(ax.shape[1]):
                    ax[i][j].axis('on')

            plot_name = 'intensity_values.png'
            plt.savefig(os.path.join(args.output_dir, fig_dir, plot_name), bbox_inches='tight')
            plt.close()

            # CSD residuals ?


            # check DWI -> T1 overlay
            for t1_file in glob(os.path.join(args.bids_dir, "sub-%s"%subject_label,
                                              "anat", "*_T1w.nii*")) + glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-*","dwi", "*_dwi.nii*")):
                if (t1_file):
                    imgT1 = nib.load(t1_file)
                    #raw = img.get_data()
                    b0_affine = img.affine
                    b0 = raw[:,:,:,shellind==0]

                    if b0.shape[3] > 0:
                        b0 = np.mean(b0, axis=3)

                    _, b0_mask = median_otsu(b0,2,1)

                    b0 = b0 * b0_mask
                    t1 = imgT1.get_data()
                    t1_affine = imgT1.affine

                    (t1, t1_affine) = helper.fixImageHeader(imgT1)

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

                    helper.plotFig(overlay, 'alignment DWI -> T1')
                    plot_name = 't1_overlay.png'
                    plt.savefig(os.path.join(args.output_dir, fig_dir, plot_name), bbox_inches='tight')
                    plt.close()
            # process & analyze outputs with python to generate all the plots

# running group level
elif args.analysis_level == "group":

    maxImg = 0
    maxList = []

    for subject_label in subjects_to_analyze:
        imgCnt = 0
        myList = []
        for image_file in glob(os.path.join(args.output_dir, 'qc_figures', "sub-%s"%subject_label, "*.png")):
            imgCnt = imgCnt + 1
            myList.extend([os.path.basename(image_file)[0:-4]])
            if maxImg < imgCnt:
                maxImg = imgCnt
                maxList = myList

    with open(os.path.join(args.output_dir, "_quality.html"), 'w') as fp:
        fp.write("<html>\n\t<body bgcolor=#FFF text=#000 style=\"font-family: Arial, Tahoma\">\n\n")
        fp.write("\t<script language=\"JavaScript\">\n\t\tfunction toggle(divClass) {\n\t\t\tclassDivs = document.getElementsByClassName(divClass);\n")
        fp.write("\t\t\tfor (var i=0; i<classDivs.length; i++) {\n\t\t\t\tdiv = classDivs[i];\n")
        fp.write("\t\t\t\tif (div.style.display === \"none\") {\n\t\t\t\t\tdiv.style.display = \"block\";\n\t\t\t\t} else {\n\t\t\t\t\tdiv.style.display = \"none\";\n\t\t\t\t}\n")
        fp.write("\t\t\t}\n\t\t}\n\t</script>\n\n")
        fp.write("\t<div id=\"menu\" style=\"width: 200px; top: 30px; z-index: 999; position: fixed; border: 1px solid gray; border-radius: 7px; padding: 10px; background-color: #EEE;\">\n")
        fp.write("\t\t<font size=5>Quality</font><p>\n")
        fp.write("\t\t<form>\n")
        for item in maxList:
            fp.write("\t\t\t<input type=\"checkbox\" onclick=\"toggle('" + str(item) + "')\" checked>" + str(item).replace('_',' ') + "</input><br>\n")
        fp.write("\t\t</form>\n<p>\n\n")
        fp.write("\t\t<div style=\"margin-left: 10px; float: right; padding: 3px; background-color: #CCC; border: 1px solid gray; border-radius: 7px;\" onclick=\"javascript: document.body.scrollTop = 0; document.documentElement.scrollTop = 0;\"><font size=2>scroll to top</font></div>\n")
        fp.write("\t</div>\n\n\t<div id=\"content\" style=\"margin-left: 230px; position: absolute; top: 10px; padding: 3px; border: 1px solid gray; border-radius: 7px; background-color: #FFF;\">\n")
        fp.write("\t\t<table>\n")

        # loop over subjects
        for subject_label in subjects_to_analyze:
            fp.write("\t\t\t<tr><td colspan=" + str(maxImg) + " bgcolor=#EEE><center><font size=3><b>sub-" + subject_label + "</b></font></center></td></tr>\n")
            # loop over images
            for image_file in glob(os.path.join(args.output_dir, 'qc_figures', "sub-%s"%subject_label, "*.png")):
                # calcualte average mask size in voxels
                fp.write("\t\t\t\t<td><div name=\"" + subject_label + "\" class=\"" + os.path.basename(image_file)[0:-4] + "\"><image src=\"" + image_file.replace(args.output_dir + os.sep, "") + "\" width=\"100%\"></div></td>\n")

        fp.write("\t\t</table>\n\t</div>\t</body>\n</html>")

    #with open(os.path.join(args.output_dir, "avg_brain_size.txt"), 'w') as fp:
    #    fp.write("Average brain size is %g voxels"%np.array(brain_sizes).mean())
