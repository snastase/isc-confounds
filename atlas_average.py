# conda activate confounds, then module load connectome_wb

from os.path import basename, exists, join
import json
import numpy as np
import nibabel as nib
from glob import glob
from gifti_io import read_gifti

space = 'fsaverage6'
atlas = 'Schaefer1000' # Schaefer1000 MMP

# Assign some directories
base_dir = '/jukebox/hasson/snastase/isc-confounds'
tpl_dir = join(base_dir, 'afni', f'tpl-{space}')

# Pick stimulus type, either narratives or movies
#task_json = join(base_dir, 'narratives_meta.json')
task_json = join(base_dir, 'movies_meta.json')

# Get metadata for all subjects for a given task
with open(task_json) as f:
    task_meta = json.load(f)


# Function to average times series within parcels
def parcel_average(data, parc, exclude_zero=True):
    labels = np.unique(parc).tolist()
    if exclude_zero:
        labels.remove(0)

    avg_data = []
    for label in labels:
        avg_data.append(np.mean(data[parc == label], axis=0))
    avg_data = np.column_stack(avg_data)

    return avg_data

# Function to expand parcel-level data back onto cortical map
def parcel_reconstruct(data, parc, n_vertices=40962,
                       exclude_zero=True, return_gii=False,
                       template_gii=None):
    labels = np.unique(parc).tolist()
    if exclude_zero:
        labels.remove(0)

    parcel_map = np.full(n_vertices, np.nan)
    for i, label in enumerate(labels):
        parcel_map[parc == label] = data[i]

    if return_gii and template_gii:
        gii = nib.load(template_gii)
        for i in np.arange(gii.numDA):
            gii.remove_gifti_data_array(0)
        gda = nib.gifti.GiftiDataArray(parcel_map.astype('float32'))
        gii.add_gifti_data_array(gda)
        parcel_map = gii

    return parcel_map

# Load in atlas
parcs, parcs_gii = {}, {}
for hemi in ['L', 'R']:
    atlas_fn = join(tpl_dir,
                    (f'tpl-fsaverage6_hemi-{hemi}_'
                     f'desc-{atlas}_dseg.label.gii'))
    parcs_gii[hemi] = nib.load(atlas_fn)
    parcs[hemi] = parcs_gii[hemi].agg_data()
    

# Sanity check visualize parcellation
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from neuromaps.transforms import fsaverage_to_fslr

# Convert fsaverage6 to fslr for visualization
parc_L = fsaverage_to_fslr(parcs_gii['L'], target_density='32k',
                           hemi='L', method='nearest')[0]
parc_R = fsaverage_to_fslr(parcs_gii['R'], target_density='32k',
                           hemi='R', method='nearest')[0]
    
surfaces_fslr = fetch_fslr()
surf_lh, surf_rh = surfaces_fslr['inflated']

p = Plot(surf_lh=surf_lh, 
         surf_rh=surf_rh,
         brightness=.7)
p.add_layer({'left': parc_L,
             'right': parc_R}, 
             cmap='YlOrRd')
fig = p.build()


# Loop through subjects and extract parcel average time series
for hemi in ['L', 'R']:
    
    for task in task_meta:
        if not exists(join(base_dir, 'afni', task)):
            mkdir(join(base_dir, 'afni', task))
        
        for subject in task_meta[task]:
            if not exists(join(base_dir, 'afni', task, subject)):
                mkdir(join(base_dir, 'afni', task, subject))

            bold_fns = task_meta[task][subject]['bold'][space]['preproc']
            bold_fns = [bold_fn for bold_fn in bold_fns
                        if f'hemi-{hemi}' in bold_fn]

            # Loop through BOLD images and extract parcel averages
            for bold_fn in bold_fns:
                bold_map = read_gifti(bold_fn)

                parc_avg = parcel_average(bold_map.T, parcs[hemi])
                assert bold_map.shape[0] == parc_avg.shape[0]

                if task in ['budapest', 'raiders']:
                    bold_fn = bold_fn.replace('task-movie',
                                              f'task-{task}')
                
                parc_1D = join(base_dir, 'afni', task, subject,
                              basename(bold_fn).replace(
                                  '_bold.func.gii',
                                  f'_parc-{atlas}_timeseries.1D'))
                np.savetxt(parc_1D, parc_avg.T, delimiter=' ',
                           newline='\n', fmt='%f')

                print(f"Extracted average {atlas} time series for "
                      f"{subject} ({task})\n  {basename(parc_1D)}")