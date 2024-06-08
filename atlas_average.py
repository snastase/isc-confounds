from os.path import join
import json
import numpy as np
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker

space = 'fsaverage6'
atlas = 'Schaefer1000'

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

def parcel_reconstruct(data, parc, n_vertices=40962,
                       exclude_zero=True):
    labels = np.unique(parc).tolist()
    if exclude_zero:
        labels.remove(0)

    parcel_map = np.full(n_vertices, np.nan)
    for i, label in enumerate(labels):
        parcel_map[parc == label] = data[i]

    return parcel_map

# Load in atlas
parcs = {}
for hemi in ['L', 'R']:
    atlas_fn = join(tpl_dir,
                    (f'tpl-fsaverage6_hemi-{hemi}_'
                    f'desc-{atlas}_dseg.label.gii'))
    parc_gii = nib.load(atlas_fn)
    parcs[hemi] = parc_gii.agg_data()
    


from surfplot import Plot
from neuromaps.datasets import fetch_fsaverage
from neuromaps.datasets import fetch_fslr
from neuromaps.transforms import mni152_to_fslr

# Fetch fsLR surfaces from neuromaps
surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
sulc_lh, sulc_rh = surfaces['sulc']

# Convert volumetric MNI data to fsLR surface
gii_lh, gii_rh = mni152_to_fslr(parcel_img, method='nearest')

# Plot example ROI on surface
p = Plot(surf_lh=lh, surf_rh=rh, brightness=.7)
p.add_layer({'left': gii_lh, 'right': gii_rh}, cmap='Blues', color_range=(0, 1))
cbar_kws = dict(location='right', draw_border=False, aspect=10,
                shrink=.2, decimals=0, pad=0, n_ticks=2)
fig = p.build(cbar_kws=cbar_kws)
