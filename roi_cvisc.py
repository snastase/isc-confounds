from os.path import join
import json
from glob import glob
from natsort import natsorted
import numpy as np
from scipy.stats import pearsonr, zscore
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

space = 'fsaverage6'
roi = 'IFG'
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

# Helper function for Fisher-transformed average
def fisher_mean(correlations, axis=None):
    return np.tanh(np.nanmean(np.arctanh(correlations), axis=axis))

# Function to compute cross-validated regression-based ISC
def cvisc(data, cv, reg=LinearRegression()):

    n_subjects = data.shape[-1]
    
    cvisc_stack = []
    for train, test in cv.split(data):
        
        # Loop through left-out subjects
        cvisc_scores = []
        for s in np.arange(n_subjects):

            # Correlation between left-out subject and mean of others
            X = zscore(np.mean(np.delete(data, s, axis=1),
                               axis=1), axis=0)[..., np.newaxis]
            y = zscore(data[..., s], axis=0)
            
            reg.fit(X[train], y[train])
            y_pred = reg.predict(X[test])
            
            score = pearsonr(y[test], y_pred)[0]
            cvisc_scores.append(score)
                    
        cvisc_stack.append(cvisc_scores)
    
    return fisher_mean(cvisc_stack, axis=0)
    
    
# Compute cvISC
cv = KFold(n_splits=2)

# Loop through tasks
tasks = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
         'notthefallintact', 'black', 'forgot']
for task in tasks:

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
            
            cviscs = cvisc(data, cv)
            
            results[model] = cviscs

            # We may want to exclude really bad EAC ISCs on principle
            n_threshold = np.sum(cviscs < threshold)

            if exclusion:
                exclude = cviscs < threshold
                cviscs = cviscs[~exclude]

            # Print mean and SD
            print(f"mean {task} ISC for model {model} = {np.mean(cviscs):.3f} "
                  f"(SD = {np.std(cviscs):.3f}) \n  {n_threshold} below {threshold} "
                  f"(exclusion = {exclusion})")

        results_fn = join(base_dir, 'results',
                          (f'results_task-{task}_roi-{roi}_'
                           f'hemi-{hemi}_desc-excl0_cviscs.npy'))
        np.save(results_fn, results)
