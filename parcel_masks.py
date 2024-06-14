# conda activate confounds

from os.path import join
import json
import numpy as np
from gifti_io import read_gifti, write_gifti_new

hemi = 'L'

# Load in HCP-MMP1 parcellation on fsaverage6
deriv_dir = '/jukebox/hasson/snastase/narratives/derivatives'
tpl_dir = join(deriv_dir, 'afni-nosmooth', 'tpl-fsaverage6')
mmp_fn = join(tpl_dir, f'tpl-fsaverage6_hemi-{hemi}_desc-MMP_dseg.label.gii')
mmp = read_gifti(mmp_fn)[0]

# Load in ROI labels
base_dir = '/jukebox/hasson/snastase/isc-confounds'
mask_dir = join(base_dir, 'afni', 'tpl-fsaverage6')
with open(join(base_dir, 'MMP_ROIs.json')) as f:
    rois = json.load(f)

roi_colors = {'EAC': rois['EAC']['A1'],
              'V1': rois['V1']['V1'],
              'IFG': rois['IFG']['45'],
              'MT+': rois['MT+']['MT']}

# Create separate masks
masks = {}
for roi in rois:
    mask = np.zeros(mmp.shape).astype('float32')
    for area in rois[roi]:
        mask[mmp == rois[roi][area]] = 1

    write_gifti_new(mask,
                join(mask_dir,
                     f'tpl-fsaverage6_hemi-{hemi}_desc-{roi}_mask.label.gii'),
                join(tpl_dir,
                     f'tpl-fsaverage6_hemi-{hemi}_desc-MMP_dseg.label.gii'))

    masks[roi] = mask.astype(bool)
    n_voxels = np.sum(mask)
    np.save(join(mask_dir,
                 f'tpl-fsaverage6_hemi-{hemi}_desc-{roi}_mask.npy'), mask)
    print(f"Created {hemi} {roi} mask containing "
          f"{n_voxels} vertices")

# Create single parcellation map
mask_map = np.zeros(mmp.shape).astype('float32')
for i, mask_name in enumerate(masks):
    mask_map[masks[mask_name]] = i + 1

write_gifti_new(mask_map,
    join(mask_dir,
         f'tpl-fsaverage6_hemi-{hemi}_desc-ROIs_dseg.label.gii'),
    join(tpl_dir,
         f'tpl-fsaverage6_hemi-{hemi}_desc-MMP_dseg.label.gii'))


# Load in Schaefer 1000 parcellation (Kong 2022) and save as GIfTI
from nibabel.freesurfer.io import read_annot

# Load in both hemispheres with labels
parc_lh, _, labels_lh = read_annot(join(mask_dir,
    'lh.Schaefer2018_1000Parcels_Kong2022_17Networks_order.annot'))
parc_rh, _, labels_rh = read_annot(join(mask_dir,
    'rh.Schaefer2018_1000Parcels_Kong2022_17Networks_order.annot'))

write_gifti_new(parc_lh,
    join(mask_dir,
         f'tpl-fsaverage6_hemi-L_desc-Schaefer1000_dseg.label.gii'),
    join(tpl_dir,
         f'tpl-fsaverage6_hemi-L_desc-MMP_dseg.label.gii'))

write_gifti_new(parc_rh,
    join(mask_dir,
         f'tpl-fsaverage6_hemi-R_desc-Schaefer1000_dseg.label.gii'),
    join(tpl_dir,
         f'tpl-fsaverage6_hemi-R_desc-MMP_dseg.label.gii'))
