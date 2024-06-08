# conda activate confounds

from os.path import join
from glob import glob
import json
from natsort import natsorted
import pandas as pd


base_dir = '/jukebox/hasson/snastase/isc-confounds'
movies_dir = '/jukebox/hasson/snastase/movies'
movies = ['budapest', 'life', 'raiders']
movie_runs = {'budapest': 5, 'life': 4, 'raiders': 4}
movie_trs = {
    'budapest': {1: 598, 2: 498, 3: 535, 4: 618, 5: 803},
    'life': {1: 374, 2: 346, 3: 377, 4: 412},
    'raiders': {1: 850, 2: 860, 3: 860, 4: 850}}

# Create a dictionary containing filenames keyed to subjects
spaces = ['fsaverage6'] # 'MNI152NLin2009cAsym'
#width = 'sm6'

movies_meta = {}
for movie in movies:
    movies_meta[movie] = {}
    
    fmriprep_dir = join(movies_dir, movie, 'derivatives',
                        'fmriprep-v23.2.3')
    subjects = natsorted([p.split('/')[-2] for p in
                      glob(join(fmriprep_dir, 'sub-*/'))])

    for subject in subjects:
        movies_meta[movie][subject] = {'bold': {},
                                       'confounds': []}
        if movie == 'raiders':
            preproc_dir = join(fmriprep_dir, subject,
                               'ses-raiders', 'func')
        else:
            preproc_dir = join(fmriprep_dir, subject, 'func')
        
        # Grab confounds TSV file        
        confound_fns = glob(join(preproc_dir,
                            f'{subject}*desc-confounds'
                            '_timeseries.tsv'))
        assert len(confound_fns) == movie_runs[movie]
        movies_meta[movie][subject]['confounds'] = \
            natsorted(confound_fns)

        for space in spaces:
            if space[:2] == 'fs':
                preproc_fns = glob(join(preproc_dir,
                    f'{subject}*space-{space}*func.gii'))
                assert len(preproc_fns) == 2 * len(confound_fns)
                
                #smooth_fns = glob(join(afni_dir, subject, 'func',
                #                       (f'{subject}*space-{space}*'
                #                        f'desc-{width}*func.gii')))
    
            #else:
            #    preproc_fns = glob(join(preproc_dir, subject, 'func',
            #                            f'{subject}*space-{space}*desc-'
            #                         'preproc_bold.nii.gz'))
            #    assert len(preproc_fns) == len(confound_fns)
    
            #    smooth_fns = glob(join(afni_dir, subject, 'func',
            #                           (f'{subject}*space-{space}*'
            #                            f'desc-{width}*_bold.nii.gz')))
    
            movies_meta[movie][subject]['bold'][space] = \
                {'preproc': natsorted(preproc_fns)} # width: smooth_fns}


# Save the subject metadata dictionary
with open(join(base_dir, 'movies_meta.json'), 'w') as f:
    json.dump(movies_meta, f, indent=2, sort_keys=True)


# Compile TRs to trim per task prior to ISC
movie_trims = {'budapest': {'start': 20, 'end': 10},
              'life': {'start': 4, 'end': 2},
              'raiders': {'start': 10, 'end': 2}} 

with open(join(base_dir, 'movies_trims.json'), 'w') as f:
    json.dump(movie_trims, f, indent=2, sort_keys=True)


# Get some summary number of runs
n_scans = []
for movie in movies:
    for s in movies_meta[movie]:
        n_scans.append(len(movies_meta[movie][s]['confounds']))
print(f"Total number of scans: {sum(n_scans)}")