# conda activate confounds

from os.path import basename, exists, join, splitext
from os import makedirs
import json
import pandas as pd
from natsort import natsorted


# Function for extracting aCompCor components
def extract_compcor(confounds_df, confounds_meta,
                    n_comps=5, method='t_comp_cor',
                    tissue=None):

    # Check that we sensible number of components
    assert n_comps > 0

    # Check that method is specified correctly
    assert method in ['a_comp_cor', 't_comp_cor',
                      'c_comp_cor', 'w_comp_cor']

    # Get CompCor metadata for relevant method
    compcor_meta = {c: confounds_meta[c] for c in confounds_meta
                    if method in c and confounds_meta[c]['Retained']}

    # Make sure metadata components are sorted properly
    comp_sorted = natsorted(compcor_meta)
    for i, comp in enumerate(comp_sorted):
        if comp != comp_sorted[-1]:
            comp_next = comp_sorted[i + 1]
            assert (compcor_meta[comp]['SingularValue'] >
                    compcor_meta[comp_next]['SingularValue'])

    # Either get top n components
    if n_comps >= 1.0:
        n_comps = int(n_comps)
        if len(comp_sorted) >= n_comps:
            comp_selector = comp_sorted[:n_comps]
        else:
            comp_selector = comp_sorted
            print(f"Warning: Only {len(comp_sorted)} {method} "
                  f"components available ({n_comps} requested)")

    # Or components necessary to capture n proportion of variance
    else:
        comp_selector = []
        for comp in comp_sorted:
            comp_selector.append(comp)
            if (compcor_meta[comp]['CumulativeVarianceExplained']
                > n_comps):
                break

    # Check we didn't end up with degenerate 0 components
    assert len(comp_selector) > 0

    # Grab the actual component time series
    confounds_compcor = confounds_df[comp_selector]

    return confounds_compcor


# Function for extracting group of (variable number) confounds
def extract_group(confounds_df, groups):
    
    # Expect list, so change if string
    if type(groups) == str:
        groups = [groups]
    
    # Filter for all columns with label
    confounds_group = []
    for group in groups:
        group_cols = [col for col in confounds_df.columns
                      if group in col]
        confounds_group.append(confounds_df[group_cols])
    confounds_group = pd.concat(confounds_group, axis=1)
    
    return confounds_group


# Function for loading in confounds files
def load_confounds(confounds_fn):

    # Load the confounds TSV files
    confounds_df = pd.read_csv(confounds_fn, sep='\t')

    # Load the JSON sidecar metadata
    with open(splitext(confounds_fn)[0] + '.json') as f:
        confounds_meta = json.load(f)

    return confounds_df, confounds_meta


# Function for extracting confounds (including CompCor)
def extract_confounds(confounds_df, confounds_meta, model_spec):

    # Pop out confound groups of variable number
    groups = set(model_spec['confounds']).intersection(
                    ['cosine', 'motion_outlier'])

    # Grab the requested confounds
    confounds = confounds_df[[c for c in model_spec['confounds']
                              if c not in groups]]
    
    # Grab confound groups if present
    if groups:
        confounds_group = extract_group(confounds_df,
                                        groups)
        confounds = pd.concat([confounds, confounds_group],
                              axis=1)

    # Get aCompCor / tCompCor confounds if requested
    compcors = set(model_spec).intersection(
                    ['a_comp_cor', 't_comp_cor',
                     'c_comp_cor', 'w_comp_cor'])
    if compcors:
        for compcor in compcors:
            if type(model_spec[compcor]) == dict:
                model_spec[compcor] = [model_spec[compcor]]

            for compcor_kws in model_spec[compcor]:
                confounds_compcor = extract_compcor(
                    confounds_df,
                    confounds_meta,
                    method=compcor,
                    **compcor_kws)

                confounds = pd.concat([confounds,
                                       confounds_compcor],
                                      axis=1)

    return confounds


# Name guard for when we want to grab all confounds
if __name__ == '__main__':

    base_dir = '/jukebox/hasson/snastase/isc-confounds'
    afni_dir = join(base_dir, 'afni')
    #tasks = ['budapest', 'life', 'raiders']
    tasks = ['pieman', 'prettymouth', 'milkyway',
             'slumlordreach', 'notthefallintact',
             'black', 'forgot']

    #task_json = join(base_dir, 'movies_meta.json')
    task_json = join(base_dir, 'narratives_meta.json')
    with open(task_json) as f:
        task_meta = json.load(f)

    with open(join(base_dir, 'model_meta.json')) as f:
        model_meta = json.load(f)
    models = model_meta.keys()

    # Loop through tasks and subjects and grab confound files
    for task in tasks:
        for subject in task_meta[task]:

            # Make directory if it doesn't exist
            ort_dir = join(afni_dir, task, subject)
            if not exists(ort_dir):
                makedirs(ort_dir)

            # Grab confound files for multiple runs if present
            confounds_fns = natsorted(
                task_meta[task][subject]['confounds'])

            # Loop through confound files (in case of multiple runs)
            for confounds_fn in confounds_fns:
                confounds_df, confounds_meta = load_confounds(confounds_fn)

                # Loop through requested models
                for model in models:

                    # Extract confounds based on model spec
                    confounds = extract_confounds(confounds_df,
                                                  confounds_meta,
                                                  model_meta[model])

                    # Create output 1D file for AFNI and save
                    ort_1d = splitext(basename(confounds_fn).replace(
                        'desc-confounds',
                        f'desc-model{model}'))[0] + '.1D'

                    if task in ['budapest', 'raiders']:
                        ort_1d = ort_1d.replace('task-movie', f'task-{task}')
                    
                    ort_fn = join(ort_dir, ort_1d)
                    confounds.to_csv(ort_fn, sep='\t', header=False,
                                     index=False)

                    # Also create CSVs with headers for convenience
                    ort_csv = splitext(basename(confounds_fn).replace(
                        'desc-confounds',
                        f'desc-model{model}'))[0] + '.csv'
                    ort_fn = join(ort_dir, ort_csv)
                    confounds.to_csv(ort_fn, sep=',', index=False)

                print(f"Assembled confound models for {subject} ({task})")
