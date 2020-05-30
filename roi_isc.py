from os.path import join
import json
from glob import glob
from natsort import natsorted
import numpy as np
from scipy.stats import zscore
from brainiak.isc import isc

task = 'forgot'
space = 'fsaverage6'
roi = 'EAC'
hemis = ['L', 'R']
threshold = .1
exclusion = False

base_dir = '/jukebox/hasson/snastase/isc-confounds'

# Get metadata for all subjects for a given task
with open(join(base_dir, 'task_meta.json')) as f:
    task_meta = json.load(f)

# Get confound models
with open(join(base_dir, 'model_meta.json')) as f:
    model_meta = json.load(f)
models = model_meta.keys()

# Load in exclusion
with open(join(base_dir, 'task_exclude.json')) as f:
    task_exclude = json.load(f)

# Load in task trims for remove bookend TRs
with open(join(base_dir, 'task_trims.json')) as f:
    task_trims = json.load(f)

# Helper function to load in AFNI's 3dTproject 1D outputs
def load_1D(fn):
    with open(fn) as f:
        lines = [line.strip().split(' ') for line in f.readlines()
                 if '#' not in line]
    assert len(lines) == 1
    return np.array(lines[0]).astype(float)
        

# Compile ROIs across all subjects
for hemi in hemis:
    results = {}
    for model in models:
        data = []
        for subject in task_meta[task]:
            
            # Check if we should a priori exclude
            if subject in task_exclude[task]:
                continue

            data_dir = join(base_dir, 'afni', subject, 'func')

            roi_fns = natsorted(glob(join(data_dir,
                          (f'{subject}_task-{task}_*space-{space}_'
                           f'hemi-{hemi}_roi-{roi}_desc-model{model}_'
                           'timeseries.1D'))))

            # Grab only first run in case of multiple runs
            roi_fn = roi_fns[0]

            # Strip comments and load in data as numpy array
            subj_data = load_1D(roi_fn)
            
            # slumlordreach has ragged end-time, so trim it
            if task == 'slumlordreach':
                subj_data = np.concatenate([zscore(subj_data[:619]),
                                            zscore(subj_data[627:1205])])
            
            data.append(zscore(subj_data))

        data = np.column_stack(data)

        # Trim data
        data = data[task_trims[task]['start']:-task_trims[task]['end'], :]

        # Compute ISCs
        iscs = isc(data).flatten()

        results[model] = iscs

        # We may want to exclude really bad EAC ISCs on principle
        n_threshold = np.sum(iscs < threshold)

        if exclusion:
            exclude = iscs < threshold
            iscs = iscs[~exclude]

        # Print mean and SD
        print(f"mean {task} ISC for model {model} = {np.mean(iscs):.3f} "
              f"(SD = {np.std(iscs):.3f}) \n  {n_threshold} below {threshold} "
              f"(exclusion = {exclusion})")

    results_fn = join(base_dir, 'results',
                      (f'results_task-{task}_roi-{roi}_'
                       f'hemi-{hemi}_desc-excl0_iscs.npy'))
    np.save(results_fn, results)
