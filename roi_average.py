# conda activate confounds

from os.path import basename, exists, join
from os import mkdir
import json
import numpy as np
from glob import glob
from gifti_io import read_gifti

space = 'fsaverage6'
roi = 'MT+' # EAC, IFG, V1, MT+

# Assign some directories
base_dir = '/jukebox/hasson/snastase/isc-confounds'
tpl_dir = join(base_dir, 'afni', f'tpl-{space}')

# Pick stimulus type, either narratives or movies
#task_json = join(base_dir, 'narratives_meta.json')
task_json = join(base_dir, 'movies_meta.json')

# Get metadata for all subjects for a given task
with open(task_json) as f:
    task_meta = json.load(f)

# Loop through subjects and extract ROI average time series
for hemi in ['L', 'R']:
    
    # Get the ROI mask
    roi_fn = join(tpl_dir, f'tpl-{space}_hemi-{hemi}_desc-{roi}_mask.npy')
    roi_mask = np.load(roi_fn)
    
    for task in task_meta:
        if not exists(join(base_dir, 'afni', task)):
            mkdir(join(base_dir, 'afni', task))
        
        for subject in task_meta[task]:
            if not exists(join(base_dir, 'afni', task, subject)):
                mkdir(join(base_dir, 'afni', task, subject))

            bold_fns = task_meta[task][subject]['bold'][space]['preproc']
            bold_fns = [bold_fn for bold_fn in bold_fns
                        if f'hemi-{hemi}' in bold_fn]

            # Loop through BOLD images and extract ROI
            for bold_fn in bold_fns:
                bold_map = read_gifti(bold_fn)
                roi_avg = np.nanmean(bold_map[:, roi_mask == 1], axis=1)
                assert bold_map.shape[0] == roi_avg.shape[0]

                if task in ['budapest', 'raiders']:
                    bold_fn = bold_fn.replace('task-movie', f'task-{task}')
                
                roi_1D = join(base_dir, 'afni', task, subject,
                              basename(bold_fn).replace(
                                  '_bold.func.gii', f'_roi-{roi}_timeseries.1D'))
                np.savetxt(roi_1D, roi_avg[None, :], delimiter=' ', fmt='%f')

                print(f"Extracted average {roi} time series for {subject} ({task})"
                       f"\n  {basename(roi_1D)}")