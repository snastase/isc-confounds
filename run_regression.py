from sys import argv
from os.path import basename, exists, join, splitext
from os import chdir
import json
from glob import glob
from subprocess import run
from natsort import natsorted

task = 'pieman'
space = 'fsaverage6'
roi = 'EAC'
smoothness = 'sm6'
models = ['model1', 'model2', 'model3', 'modelX']

# Check that we get a reasonable space
if 'MNI' not in space and 'fsaverage' not in space:
    raise AssertionError("Expected either MNI or fsaverage space")
    
# Assign some directories
base_dir = '/jukebox/hasson/snastase/isc-confounds'
deriv_dir = '/jukebox/hasson/snastase/narratives/derivatives'

# Get metadata for all subjects for a given task
with open(join(base_dir, 'task_meta.json')) as f:
    task_meta = json.load(f)
subjects = task_meta[task].keys()

# Get the volumetric mask resampled for this task
if 'fsaverage' not in space:
    mask_fn = join(deriv_dir, 'afni', f'tpl-{space}',
                   f'tpl-{space}_res-{task}_desc-brain_mask.nii.gz')
    assert exists(mask_fn)

# Loop through requested models
for model in models:
    
    # Loop through requested subjects
    for subject in subjects:
        afni_dir = join(deriv_dir, 'afni', subject, 'func')
        confounds_dir = join(base_dir, 'afni', subject, 'func')
        
        # Move into output directory (AFNI seems to want to save in wd)
        chdir(confounds_dir)

        # Get input BOLD data
        if roi:
            bold_fns = natsorted(glob(join(confounds_dir,
                           (f'{subject}_task-{task}_*space-{space}_'
                            f'hemi-*_desc-{roi}_timeseries.1D'))))
        elif 'fsaverage' in space:
            bold_fns = natsorted(glob(join(afni_dir,
                           (f'{subject}_task-{task}_*space-{space}_'
                            f'hemi-*_desc-{smoothness}.func.gii'))))
        else:
            bold_fns = natsorted(glob(join(afni_dir,
                           (f'{subject}_task-{task}_*space-{space}_'
                            f'hemi-{hemi}_desc-{smoothness}_bold.nii.gz'))))
        
        # Loop through multiple input files (e.g. runs or hemispheres)
        for bold_fn in bold_fns:
            
            ort_fn = (basename(bold_fn).split('space')[0] +
                      f'desc-{model}_regressors.1D')

            # Assign regression output filename (i.e. residuals)
            if roi:
                reg_fn = join(confounds_dir,
                              basename(bold_fn).replace(
                                  f'desc-{roi}',
                                  f'roi-{roi}_desc-{model}'))
            else:
                reg_fn = join(confounds_dir,
                              basename(bold_fn).replace(f'desc-{smoothness}',
                                                        f'desc-{model}'))

            # Run AFNI's 3dTproject
            if 'fsaverage' in space:
                run(f"3dTproject -input {bold_fn} -ort {ort_fn} "
                    f"-prefix {reg_fn} -polort 2", shell=True)
            else:
                run(f"3dTproject -input {bold_fn} -ort {ort_fn} "
                    f"-prefix {reg_fn} -mask {mask} -polort 2", shell=True)

            print(f"Finished {model} regression for {subject} ({task}): "
                  f"\n  {basename(reg_fn)}")