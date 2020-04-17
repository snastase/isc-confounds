from os.path import join
from glob import glob
import json
from natsort import natsorted


base_dir = '/jukebox/hasson/snastase/isc-confounds'
data_dir = '/jukebox/hasson/snastase/narratives/derivatives/fmriprep'


# Get our subject list from the BIDS directory
n_subjects = 345
subjects = natsorted([p.split('/')[-2] for p in
                      glob(join(data_dir, 'sub-*/'))])

# Check that we're not missing subjects
for n, subject in zip(range(1, n_subjects + 1), subjects):
    if subject != f"sub-{n:03}":
        raise AssertionError("Found a mismatching subject: "
                             f"{subject} (expected sub-{n:03})")

# Create a dictionary containing filenames keyed to subjects
spaces = ['MNI152NLin2009cAsym', 'fsaverage6']
subject_meta = {}
for subject in subjects:
    subject_meta[subject] = {'bold': {},
                             'confounds': []}

    # Grab confounds TSV file
    confound_fns = glob(join(data_dir, subject, 'func',
                             f'{subject}*desc-confounds'
                             '_regressors.tsv'))
    subject_meta[subject]['confounds'] = confound_fns

    # Grab either volumetric or surface-based BOLD filenames
    for space in spaces:
        if space[:2] == 'fs':
            bold_fns = glob(join(data_dir, subject, 'func',
                                 f'{subject}*space-{space}*func.gii'))
            assert len(bold_fns) == 2 * len(confound_fns)
        else:
            bold_fns = glob(join(data_dir, subject, 'func',
                                 f'{subject}*space-{space}*desc-'
                                 'preproc_bold.nii.gz'))
            assert len(bold_fns) == len(confound_fns)

        subject_meta[subject]['bold'][space] = bold_fns

# Save the subject metadata dictionary
with open(join(base_dir, 'subject_meta.json'), 'w') as f:
    json.dump(subject_meta, f, indent=2, sort_keys=True)


# Compile task list from BIDS data
tasks = []
for subject in subject_meta:
    for confound_fn in subject_meta[subject]['confounds']:
        task = confound_fn.split('task-')[-1].split('_')[0]
        if task not in tasks:
            tasks.append(task)

tasks = natsorted(tasks)

# Create a dictionary of filenames keyed to tasks for convenience
task_meta = {}
for task in tasks:
    task_meta[task] = {}
    for subject in subject_meta:

        # Check that subject received task and setup dictionary
        has_task = False
        for confound_fn in subject_meta[subject]['confounds']:
            if f"task-{task}_" in confound_fn:
                task_meta[task][subject] = {'bold': {},
                                            'confounds': []}
                has_task = True

        # Ugly redundant loop but c'est la vie!
        if has_task:
            for confound_fn in subject_meta[subject]['confounds']:
                if f"task-{task}_" in confound_fn:
                    task_meta[task][subject]['confounds'].append(
                        confound_fn)
            for space in subject_meta[subject]['bold']:
                task_meta[task][subject]['bold'][space] = []
                for bold_fn in subject_meta[subject]['bold'][space]:
                    if f"task-{task}_" in bold_fn:
                        task_meta[task][subject]['bold'][space].append(
                            bold_fn)

# Save the task metadata dictionary
with open(join(base_dir, 'task_meta.json'), 'w') as f:
    json.dump(task_meta, f, indent=2, sort_keys=True)

# Get some summary number of runs
n_scans = sum([len(subject_meta[s]['confounds'])
               for s in subject_meta])

# Check that task metadata matches up
n_scans_task = []
for task in task_meta:
    for subject in task_meta[task]:
        n_scans_task.append(
            len(task_meta[task][subject]['confounds']))
n_scans_task = sum(n_scans_task)

assert n_scans == n_scans_task
print(f"Total number of scans: {n_scans}")
