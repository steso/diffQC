## Diffusion Quality Control (diffQC) BIDS App

### Development
User this command to build/rebuild docker container:
```sh
	docker build -t diff_qc diffQC/
```
Run participant-level analysis with e.g. testbench bids-dataset:
```sh
docker run -ti --rm \
	-v ~/polybox/bids_testbench:/input \
	-v ~/polybox/bids_test_out:/output \
	diff_qc \
	/input /output participant
```
### Description
This BIDS-app performs quality estimations of MRI Diffusion datasets. It is still under development, please report any issues.

### Documentation
So far denoising and dwi2tensor is performed in [MRtrix3](http://www.mrtrix.org).

### How to report errors
Provide instructions for users on how to get help and report errors.

### Acknowledgements
When using diffQC, please use cite the following work:

denoising -> Veraart2016
Gibbs ringing removal (Kellner et al., 2016)
dwi2tensor -> Veraart 2014
```
Kellner, E.; Dhital, B.; Kiselev, V. G.; Reisert, M. Gibbs-ringing artifact removal based on local subvoxel-shifts. Magnetic Resonance in Medicine, 2006, 76(5), 1574-1581
Veraart, J.; Sijbers, J.; Sunaert, S.; Leemans, A. & Jeurissen, B. Weighted linear least squares estimation of diffusion MRI parameters: strengths, limitations, and pitfalls. NeuroImage, 2013, 81, 335-346
Veraart, J.; Fieremans, E. & Novikov, D.S. Diffusion MRI noise mapping using random matrix theory Magn. Res. Med., 2016, early view, doi:10.1002/mrm.26059
```

### Usage
The quality examination requires the data to be organzied according to the [BIDS specifications](http://bids.neuroimaging.io/).
This App has the following command line arguments:

		usage: run.py [-h]
		              [--participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]]
		              bids_dir output_dir {participant,group}

		Example BIDS App entry point script.

		positional arguments:
		  bids_dir              The directory with the input dataset formatted
		                        according to the BIDS standard.
		  output_dir            The directory where the output files should be stored.
		                        If you are running a group level analysis, this folder
		                        should be prepopulated with the results of
		                        the participant level analysis.
		  {participant,group}   Level of the analysis that will be performed. Multiple
		                        participant level analyses can be run independently
		                        (in parallel).

		optional arguments:
		  -h, --help            show this help message and exit
		  --participant_label PARTICIPANT_LABEL [PARTICIPANT_LABEL ...]
		                        The label(s) of the participant(s) that should be
		                        analyzed. The label corresponds to
		                        sub-<participant_label> from the BIDS spec (so it does
		                        not include "sub-"). If this parameter is not provided
		                        all subjects will be analyzed. Multiple participants
		                        can be specified with a space separated list.

To run it in participant level mode (for one participant):

    docker run -i --rm \
		-v /Users/filo/data/ds005:/bids_dataset:ro \
		-v /Users/filo/outputs:/outputs \
		bids/example \
		/bids_dataset /outputs participant --participant_label 01

After doing this for all subjects (potentially in parallel), the group level analysis
can be run:

    docker run -i --rm \
		-v /Users/filo/data/ds005:/bids_dataset:ro \
		-v /Users/filo/outputs:/outputs \
		bids/example \
		/bids_dataset /outputs group

### Special considerations
Describe whether your app has any special requirements. For example:

- Multiple map reduce steps (participant, group, participant2, group2 etc.)
- Unusual memory requirements
- etc.
