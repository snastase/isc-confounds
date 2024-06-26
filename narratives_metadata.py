# conda activate confounds

from os.path import join
from glob import glob
import json
from natsort import natsorted
import pandas as pd


base_dir = '/jukebox/hasson/snastase/isc-confounds'
narratives_dir = '/jukebox/hasson/snastase/narratives'
preproc_dir = join(narratives_dir, 'derivatives', 'fmriprep-v23.2.3')
#afni_dir = join(narratives_dir, 'derivatives', 'afni')


# Get our subject list from the BIDS directory
n_subjects = 345
#subjects = natsorted([p.split('/')[-2] for p in
#                      glob(join(preproc_dir, 'sub-*/'))])
subjects = natsorted([p.split('/')[-1][:7] for p in
                      glob(join(preproc_dir, 'sub-*.html'))])

# Check that we're not missing subjects
for n in range(1, n_subjects + 1):
    if f"sub-{n:03}" not in subjects:
        #raise AssertionError("Found a mismatching subject: "
        #                     f"{subject} (expected sub-{n:03})")
        print("Found a missing subject: "
              f"{n} (expected sub-{n:03})")

# Create a dictionary containing filenames keyed to subjects
spaces = ['MNI152NLin2009cAsym', 'fsaverage6']
width = 'sm6'
subject_meta = {}
for subject in subjects:
    subject_meta[subject] = {'bold': {},
                             'confounds': []}

    # Grab confounds TSV file
    confound_fns = glob(join(preproc_dir, subject, 'func',
                             f'{subject}*desc-confounds'
                             '_timeseries.tsv'))
    subject_meta[subject]['confounds'] = confound_fns

    # Grab either volumetric or surface-based BOLD filenames
    for space in spaces:
        if space[:2] == 'fs':
            preproc_fns = glob(join(preproc_dir, subject, 'func',
                                    f'{subject}*space-{space}*func.gii'))
            assert len(preproc_fns) == 2 * len(confound_fns)
            
            #smooth_fns = glob(join(afni_dir, subject, 'func',
            #                       (f'{subject}*space-{space}*'
            #                        f'desc-{width}*func.gii')))

        else:
            preproc_fns = glob(join(preproc_dir, subject, 'func',
                                    f'{subject}*space-{space}*desc-'
                                 'preproc_bold.nii.gz'))
            assert len(preproc_fns) == len(confound_fns)

            #smooth_fns = glob(join(afni_dir, subject, 'func',
            #                       (f'{subject}*space-{space}*'
            #                        f'desc-{width}*_bold.nii.gz')))

        subject_meta[subject]['bold'][space] = {'preproc': preproc_fns}
                                                #width: smooth_fns}

# Save the subject metadata dictionary
with open(join(base_dir, 'narratives_subject_meta.json'), 'w') as f:
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
descs = ['preproc']#, 'sm6']
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
                task_meta[task][subject]['bold'][space] = {}
                for desc in descs:
                    task_meta[task][subject]['bold'][space][desc] = []
                    for bold_fn in subject_meta[subject]['bold'][space][desc]:
                        if f"task-{task}_" in bold_fn:
                            task_meta[task][subject]['bold'][space][
                                desc].append(bold_fn)

# Save the task metadata dictionary
with open(join(base_dir, 'narratives_meta_all.json'), 'w') as f:
    json.dump(task_meta, f, indent=2, sort_keys=True)
    

# Set up some subjects to exclude a priori due to
# e.g. mismatching TRs, poor behavior, noncompliance
task_exclude = {'pieman': [],
                'prettymouth': ['sub-038', 'sub-105'],
                'milkyway': ['subj-105', 'sub-123', 'sub-038'],
                'slumlordreach': ['sub-139'],
                'notthefallintact': [],
                'black': [],
                'forgot': []}

with open(join(base_dir, 'task_exclude.json'), 'w') as f:
    json.dump(task_exclude, f, indent=2, sort_keys=True)


# Filter for the tasks / conditions we care about in confound analysis
tasks = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
         'notthefallintact', 'black', 'forgot']
participants_fn = join(narratives_dir, 'participants.tsv')
df = pd.read_csv(participants_fn, sep='\t')

all_tasks = list(task_meta.keys())

for task in all_tasks:
    
    if task not in tasks:
        del task_meta[task]
        continue
    
    if task == 'prettymouth':
        keep_ids = df['participant_id'][(df['task'].str.contains('prettymouth')) &
                                        (df['condition'].str.contains('affair'))]
    elif task == 'milkyway':
        keep_ids = df['participant_id'][(df['task'].str.contains('milkyway')) &
                                      (df['condition'].str.contains('original'))]
    else:
        keep_ids = task_meta[task].keys()
    keep_ids = list(keep_ids)
    
    participants = list(task_meta[task].keys())
    for p in participants:
        if p not in keep_ids:
            del task_meta[task][p]

# Save the filtered task metadata for confound models
with open(join(base_dir, 'narratives_meta.json'), 'w') as f:
    json.dump(task_meta, f, indent=2, sort_keys=True)


# Compile TRs to trim per task prior to ISC
task_trims = {'pieman': {'start': 10, 'end': 8},
              'prettymouth': {'start': 14, 'end': 10},
              'milkyway': {'start': 14, 'end': 10},
              'slumlordreach': {'start': 20, 'end': 8},
              'notthefallintact': {'start': 2, 'end': 8},
              'black': {'start': 8, 'end': 8},
              'forgot': {'start': 8, 'end': 8}}

with open(join(base_dir, 'narratives_trims.json'), 'w') as f:
    json.dump(task_trims, f, indent=2, sort_keys=True)


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

#assert n_scans == n_scans_task
print(f"Total number of scans: {n_scans}")
