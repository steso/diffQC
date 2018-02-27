#!/usr/bin/env python3
import argparse
import os
import subprocess
import nibabel
import numpy
from glob import glob
from mrtrix3 import app, file, fsl, image, path, run

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
        subject_dir = os.path.join(args.output_dir, 'sub-' + subject_label)
        if not os.path.isdir(subject_dir):
            os.makedirs(subject_dir)

        for dwi_file in glob(os.path.join(args.bids_dir, "sub-%s"%subject_label,
                                          "dwi", "*_dwi.nii*")) + glob(os.path.join(args.bids_dir,"sub-%s"%subject_label,"ses-*","dwi", "*_dwi.nii*")):

            # Step 1: Denoise
            # run.command('dwidenoise ' + dwidenoise_input + ' dwi_denoised.' + ('nii' if unring_cmd else 'mif'))
            # if unring_cmd:
            #   run.command('mrinfo ' + dwidenoise_input + ' -json_keyval input.json')
            # file.delTemporary(dwidenoise_input)
            #
            # # Step 2: Gibbs ringing removal (if available)
            # if unring_cmd:
            #     run.command(unring_cmd + ' dwi_denoised.nii dwi_unring' + fsl_suffix + ' -n 100')
            #     file.delTemporary('dwi_denoised.nii')
            #     unring_output_path = fsl.findImage('dwi_unring')
            #     run.command('mrconvert ' + unring_output_path + ' dwi_unring.mif -json_import input.json')
            #     file.delTemporary(unring_output_path)
            #     file.delTemporary('input.json')

            # Denoising to obtain noise-map
            out_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_denoised.")
            noise_file = os.path.split(dwi_file)[-1].replace("_dwi.", "_noise.")
            cmd = "dwidenoise %s %s -noise %s -force"%(dwi_file,
                                                       os.path.join(args.output_dir, subject_dir, out_file),
                                                       os.path.join(args.output_dir, subject_dir, noise_file))
            run(cmd)

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

            # CSD residuals ?



            # process & analyze outputs with python to generate all the plots

# running group level
elif args.analysis_level == "group":
    brain_sizes = []
    for subject_label in subjects_to_analyze:
        for brain_file in glob(os.path.join(args.output_dir, "sub-%s*.nii*"%subject_label)):
            data = nibabel.load(brain_file).get_data()
            # calcualte average mask size in voxels
            brain_sizes.append((data != 0).sum())

    with open(os.path.join(args.output_dir, "avg_brain_size.txt"), 'w') as fp:
        fp.write("Average brain size is %g voxels"%numpy.array(brain_sizes).mean())
