from os.path import join
import json
from natsort import natsorted
import numpy as np
from extract_confounds import extract_confounds, load_confounds
from brainiak.isc import isc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

base_dir = '/jukebox/hasson/snastase/isc-confounds'
results_dir = join(base_dir, 'results')

with open(join(base_dir, 'task_meta.json')) as f:
    task_meta = json.load(f)

# Load in exclusion
with open(join(base_dir, 'task_exclude.json')) as f:
    task_exclude = json.load(f)


# Loop through tasks and subjects and grab FDs
fds = {}
for task in task_meta:
    fds[task] = {}
    for subject in task_meta[task]:

        # Only grab files for first run to match ISC
        confounds_fn = natsorted(task_meta[task][subject]['confounds'])[0]

        confounds_df, confounds_meta = load_confounds(confounds_fn)

        # Extract confounds based on model spec
        subj_fd = extract_confounds(confounds_df,
                                    confounds_meta,
                                    {'confounds':
                                     ['framewise_displacement']})
        fds[task][subject] = subj_fd.to_numpy()

        print(f"Extracted framewise displacement for {task} {subject}")

np.save(join(results_dir, 'task_fd.npy'), fds)


# Compute ISCs on framewise displacement
fd_iscs = {}
for task in task_meta:
    task_fd = []
    for subject in task_meta[task]:

        # Check if we should a priori exclude
        if subject in task_exclude[task]:
            continue

        # Get framewise displacement
        subj_fd = fds[task][subject]

        # Trim ragged-end
        if task == 'slumlordreach':
            subj_fd = subj_fd[:1205]

        task_fd.append(subj_fd)

    task_fd = np.column_stack(task_fd)

    # Compute ISCs
    iscs = isc(task_fd[1:, :])[:, 0]
    fd_iscs[task] = iscs


# Plot FD ISCs
task_dfs = []
for task in task_meta:
    task_df = pd.DataFrame(fd_iscs[task])
    task_df.rename(columns={0: 'FD ISC'}, inplace=True)
    task_df['task'] = task
    task_dfs.append(task_df)

fd_df = pd.concat(task_dfs)

task_order = ['pieman', 'prettymouth', 'milkyway', 
              'slumlordreach', 'notthefallintact',
              'black', 'forgot']

fig, ax = plt.subplots(figsize=(6, 4))
sns.stripplot(x='task', y='FD ISC', data=fd_df,
              color='.8', zorder=0, ax=ax)
sns.pointplot(x='task', y='FD ISC', data=fd_df, color='darkred',
              order=task_order, join=False, ax=ax)
ax.axhline(0, color='gray', linestyle='--', zorder=0)
ax.set_xticklabels(task_order, rotation=90)


# Correlate mean FD with ISC
task = 'black'
roi = 'EAC'
hemi = 'L'
model = '0'

iscs_fn = join(base_dir, 'results',
               (f'results_task-{task}_roi-{roi}_'
                f'hemi-{hemi}_desc-excl0_iscs.npy'))
iscs = np.load(iscs_fn, allow_pickle=True).item()[model]
mean_fds = [np.mean(fds[task][s][1:]) for s in fds[task]]
assert len(iscs) == len(mean_fds)

r, p = pearsonr(iscs, mean_fds)
print(f"Correlation between {task} FD and ISC = {r:.3f} (p = {p:.3f}) ")
sns.regplot(mean_fds, iscs)

