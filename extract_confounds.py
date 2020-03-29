from os.path import basename, join, splitext
import json
import pandas as pd
from natsort import natsorted

base_dir = '/jukebox/hasson/snastase/isc-confounds'
data_dir = '/jukebox/hasson/snastase/narratives/derivatives/fmriprep'
confounds_fn = join(data_dir, 'sub-001', 'func',
                    'sub-001_task-pieman_run-1_desc-confounds_regressors.tsv') 


# Function for extracting aCompCor components
def extract_compcor(confounds_df, confounds_meta,
                    n_comps=5, method='tCompCor',
                    tissue=None):

    # Check that we sensible number of components
    assert n_comps > 0

    # Check that method is specified correctly
    assert method in ['aCompCor', 'tCompCor']

    # Check that tissue is specified for aCompCor
    if method == 'aCompCor' and tissue not in ['combined', 'CSF', 'WM']:
        raise AssertionError("Must specify a tissue type "
                             "(combined, CSF, or WM) for aCompCor")

    # Ignore tissue if specified for tCompCor
    if method == 'tCompCor' and tissue:
        print("Warning: tCompCor is not restricted to a tissue "
              f"mask - ignoring tissue specification ({tissue})")
        tissue = None

    # Get CompCor metadata for relevant method
    compcor_meta = {c: confounds_meta[c] for c in confounds_meta
                    if confounds_meta[c]['Method'] == method
                    and confounds_meta[c]['Retained']}

    # If aCompCor, filter metadata for tissue mask
    if method == 'aCompCor':
        compcor_meta = {c: compcor_meta[c] for c in compcor_meta
                        if compcor_meta[c]['Mask'] == tissue}

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
        if len(comp_sorted) > n_comps:
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


# Function for loading in confounds files
def load_confounds(input_fn):

    # Load the confounds TSV files
    confounds_df = pd.read_csv(confounds_fn, sep='\t')

    # Load the JSON sidecar metadata
    with open(splitext(confounds_fn)[0] + '.json') as f:
        confounds_meta = json.load(f)

    return confounds_df, confounds_meta


# Function for saving confounds for AFNI 3dTproject (-ort)
def save_confounds(output_fn, confounds):
    confounds.to_csv(output_fn, sep='\t',
                     header=False, index=False)


# Function for extracting confounds (including CompCor)
def extract_confounds(confounds_df, confounds_meta, confound_labels,
                      acompcor_kws=None, tcompcor_kws=None):

    # Check that the specified confounds are in the dataframe
    for label in confound_labels:
        if label not in confounds_df:
            raise AssertionError(f"Confound {label} was "
                                 "not found")

    # Grab the requested confounds
    confounds = confounds_df[confounds]

    return confounds


# Name guard for when we actually want to split all data
if __name__ == '__main__':
    pass