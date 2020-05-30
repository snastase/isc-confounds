from os.path import basename, join
import json
import numpy as np
from glob import glob
from gifti_io import read_gifti

space = 'fsaverage6'
roi = 'EAC'

# Assign some directories
base_dir = '/jukebox/hasson/snastase/isc-confounds'
deriv_dir = '/jukebox/hasson/snastase/narratives/derivatives'
tpl_dir = join(deriv_dir, 'afni', f'tpl-{space}')

# Get metadata for all subjects for a given task
with open(join(base_dir, 'task_meta.json')) as f:
    task_meta = json.load(f)

# Loop through subjects and extract ROI average time series
for hemi in ['L', 'R']:
    
    # Get the ROI mask
    roi_fn = join(tpl_dir, f'tpl-{space}_hemi-{hemi}_desc-{roi}_mask.npy')
    roi_mask = np.load(roi_fn)
    
    for task in task_meta:
        for subject in task_meta[task]:   
            confounds_dir = join(base_dir, 'afni', subject, 'func')

            bold_fns = task_meta[task][subject]['bold'][space]['preproc']
            bold_fns = [bold_fn for bold_fn in bold_fns
                        if f'hemi-{hemi}' in bold_fn]

            # Loop through BOLD images and extract ROI
            for bold_fn in bold_fns:
                bold_map = read_gifti(bold_fn)
                roi_avg = np.nanmean(bold_map[:, roi_mask == 1], axis=1)
                assert bold_map.shape[0] == roi_avg.shape[0]

                roi_1D = join(base_dir, 'afni', subject, 'func',
                              basename(bold_fn).replace(
                                  '.func.gii', f'_desc-{roi}_timeseries.1D'))
                np.savetxt(roi_1D, roi_avg[None, :], delimiter=' ', fmt='%f')

                print(f"Extracted average {roi} time series for {subject} ({task})"
                       f"\n  {basename(roi_1D)}")