# conda activate brainiak

from os.path import join
import json
from glob import glob
from natsort import natsorted
import numpy as np
from scipy.stats import zscore
from brainiak.isc import isc

space = 'fsaverage6'
parc = 'IFG' # V1 MT+ EAC IFG
#parc = 'Schaefer1000'
hemis = ['L', 'R']
threshold = .1
exclusion = False

base_dir = '/jukebox/hasson/snastase/isc-confounds'

# Get metadata for all subjects for a given task
task_json = join(base_dir, 'narratives_meta.json')
trims_json = join(base_dir, 'narratives_trims.json')
exclude_json = join(base_dir, 'narratives_exclude.json')
#task_json = join(base_dir, 'movies_meta.json')
#trims_json = join(base_dir, 'movies_trims.json')
#exclude_json = join(base_dir, 'movies_exclude.json')

# Set up for multiple runs per task
tasks = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
         'notthefallintact', 'black', 'forgot']
task_runs = {t: [None] for t in tasks}

#task_runs = {'budapest': [1, 2, 3, 4, 5],
#             'life': [1, 2, 3, 4],
#             'raiders': [1, 2, 3, 4]}

with open(task_json) as f:
    task_meta = json.load(f)
    
# Load in task trims for remove bookend TRs
with open(trims_json) as f:
    task_trims = json.load(f)
    
# Load in exclusion
with open(exclude_json) as f:
    task_exclude = json.load(f)
    
# Get confound models
with open(join(base_dir, 'model_meta.json')) as f:
    model_meta = json.load(f)
models = model_meta.keys()

# Helper function to load in AFNI's 3dTproject 1D outputs
def load_1D(fn):
    with open(fn) as f:
        lines = [line.strip().split(' ') for line in f.readlines()
                 if '#' not in line]
    #assert len(lines) == 1
    return np.array(lines).astype(float).T
        

# Compile ROIs across all subjects
# Loop through tasks
for task in task_runs:

    for run in task_runs[task]:
    
        for hemi in hemis:
            results = {}
            for model in models:
                data = []
                for subject in task_meta[task]:

                    # Check if we should a priori exclude
                    if (task not in ['budapest', 'life', 'raiders']
                        and subject in task_exclude[task]):
                        continue
                        
                    if (task in task_exclude
                        and subject in task_exclude[task]
                        and run in task_exclude[task][subject]):
                        continue

                    data_dir = join(base_dir, 'afni', task, subject)

                    if run:
                        parc_fns = natsorted(glob(join(data_dir,
                                      (f'{subject}*task-{task}_*run-{run:02d}_'
                                       f'hemi-{hemi}_space-{space}_'
                                       f'parc-{parc}_desc-model{model}_'
                                       'timeseries.1D'))))
                        raise
                    else:
                        parc_fns = natsorted(glob(join(data_dir,
                                      (f'{subject}*task-{task}_*'
                                       f'hemi-{hemi}_space-{space}_'
                                       f'parc-{parc}_desc-model{model}_'
                                       'timeseries.1D'))))                        

                    # Grab only first run in case of multiple runs
                    if len(parc_fns) == 0:
                        print(f"Skipping subject {subject}, model {model},"
                              f"hemisphere {hemi}, task {task}")
                        print("Maybe too many nuisance regressors?!")
                    else:
                        parc_fn = parc_fns[0]

                    # Strip comments and load in data as numpy array
                    subj_data = load_1D(parc_fn)

                    # slumlordreach has ragged end-time, so trim it
                    if task == 'slumlordreach':
                        subj_data = np.concatenate([zscore(subj_data[:619]),
                                                    zscore(subj_data[627:1205])])

                    data.append(zscore(subj_data))

                # Check whether single ROI or parcellation data
                if data[0].ndim > 1:
                    data = np.stack(data, -1)
                else:
                    data = np.column_stack(data)

                # Trim data
                data = data[task_trims[task]['start']:-task_trims[task]['end']]

                # Compute ISCs
                iscs = isc(data)
                
                if iscs.shape[1] == 1:
                    iscs = iscs.squeeze()

                results[model] = iscs

                # We may want to exclude really bad EAC ISCs on principle
                #n_threshold = np.sum(iscs < threshold)

                if exclusion:
                    exclude = iscs < threshold
                    iscs = iscs[~exclude]

                # Print mean and SD
                print(f"mean {task} ISC for model {model} = {np.nanmean(iscs):.3f} "
                      f"(SD = {np.nanstd(iscs):.3f})")

            if run:
                results_fn = join(base_dir, 'results',
                                  (f'results_task-{task}_run-{run:02d}_'
                                   f'hemi-{hemi}_parc-{parc}_iscs.npy'))
            else:
                results_fn = join(base_dir, 'results',
                                  (f'results_task-{task}_'
                                   f'hemi-{hemi}_parc-{parc}_iscs.npy'))
            np.save(results_fn, results)
