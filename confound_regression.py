# conda activate confounds, then module load afni/2023.10.12

from sys import argv
from os.path import basename, exists, join, splitext
from os import chdir
import json
from glob import glob
from subprocess import run
from natsort import natsorted
from shutil import copyfile

space = 'fsaverage6'
# smoothness = 'sm6'

# Check that we get a reasonable space
if 'MNI' not in space and 'fsaverage' not in space:
    raise AssertionError("Expected either MNI or fsaverage space")
    
# Assign some directories
base_dir = '/jukebox/hasson/snastase/isc-confounds'
#deriv_dir = '/jukebox/hasson/snastase/narratives/derivatives'

# Set stimulus type and ROI
#task_json = join(base_dir, 'narratives_meta.json')
task_json = join(base_dir, 'movies_meta.json')
#parc_avg = 'MT+' # V1 MT+ EAC IFG
parc_avg = 'Schaefer1000'

# Get metadata for all subjects for a given task
with open(task_json) as f:
    task_meta = json.load(f)

# Get confound models
with open(join(base_dir, 'model_meta.json')) as f:
    model_meta = json.load(f)
models = model_meta.keys()

# Get the volumetric mask resampled for this task
if 'fsaverage' not in space:
    mask_fn = join(deriv_dir, 'afni', f'tpl-{space}',
                   f'tpl-{space}_res-{task}_desc-brain_mask.nii.gz')
    assert exists(mask_fn)
    
# Loop through tasks
#tasks = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
#         'notthefallintact', 'black', 'forgot']
tasks = ['budapest', 'life', 'raiders']
for task in tasks:
    
    # Loop through requested models
    for model in models:
    
        # Loop through requested subjects
        for subject in task_meta[task]:

            #afni_dir = join(deriv_dir, 'afni', task, subject)
            confounds_dir = join(base_dir, 'afni', task, subject)

            # Move into output directory (AFNI seems to want to save in wd)
            chdir(confounds_dir)

            # Get input BOLD data
            if parc_avg:
                bold_fns = natsorted(glob(join(confounds_dir,
                               (f'{subject}*task-{task}_*_space-{space}_'
                                f'parc-{parc_avg}_timeseries.1D'))))
                assert len(bold_fns) > 0

            #elif 'fsaverage' in space:
            #    bold_fns = natsorted(glob(join(afni_dir,
            #                   (f'{subject}_task-{task}_*space-{space}_'
            #                    f'hemi-*_desc-{smoothness}.func.gii'))))
            #else:
            #    bold_fns = natsorted(glob(join(afni_dir,
            #                   (f'{subject}_task-{task}_*space-{space}_'
            #                    f'res-native_desc-{smoothness}_bold.nii.gz'))))

            # Loop through multiple input files (e.g. runs or hemispheres)
            for bold_fn in bold_fns:

                # Assign regression output filename (i.e. residuals)
                if parc_avg:
                    reg_fn = join(confounds_dir,
                                  basename(bold_fn).replace(
                                      f'parc-{parc_avg}',
                                      f'parc-{parc_avg}_desc-model{model}'))
                #else:
                #    if f'desc-{smoothness}' in bold_fn:
                #        reg_fn = join(confounds_dir,
                #                      basename(bold_fn).replace(
                #                          f'desc-{smoothness}',
                #                          f'desc-model{model}'))
                #    else:
                #        raise AssertionError(f"Unrecogized filename {bold_fn}")

                # Simply copy/rename model 0 (no confound regression)
                if model == '0':
                    run(f"3dTproject -input {bold_fn} "
                            f"-prefix {reg_fn} -polort 2 -overwrite",
                        shell=True)

                # Otherwise perform confound regression
                else:
                    ort_fn = (basename(bold_fn).split('hemi')[0] +
                              f'desc-model{model}_timeseries.1D')

                    # Run AFNI's 3dTproject
                    if 'fsaverage' in space:
                        run(f"3dTproject -input {bold_fn} -ort {ort_fn} "
                            f"-prefix {reg_fn} -polort 2 -overwrite", shell=True)
                    else:
                        run(f"3dTproject -input {bold_fn} -ort {ort_fn} -overwrite "
                            f"-prefix {reg_fn} -mask {mask} -polort 2", shell=True)

                    print(f"Finished model {model} regression for {subject} "
                          f"({task}):\n  {basename(reg_fn)}")
