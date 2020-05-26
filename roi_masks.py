from os.path import join
import json
import numpy as np
from gifti_io import read_gifti, write_gifti

hemi = 'R'

# Load in HCP-MMP1 parcellation on fsaverage6
deriv_dir = '/jukebox/hasson/snastase/narratives/derivatives'
tpl_dir = join(deriv_dir, 'afni', 'tpl-fsaverage6')
mmp_fn = join(tpl_dir, f'tpl-fsaverage6_hemi-{hemi}_desc-MMP_dseg.label.gii')
mmp = read_gifti(mmp_fn)[0]

# Load in ROI labels
base_dir = '/jukebox/hasson/snastase/isc-confounds'
with open(join(base_dir, 'MMP_ROIs.json')) as f:
    rois = json.load(f)

roi_colors = {'EAC': rois['EAC']['A1'],
              'V1': rois['EVC']['V1']}

rois['V1'] = {'V1': 1}
rois.pop('EVC')

# Create separate masks
masks = {}
for roi in rois:
    mask = np.zeros(mmp.shape)
    for area in rois[roi]:
        mask[mmp == rois[roi][area]] = 1

    write_gifti(mask,
                join(tpl_dir,
                     f'tpl-fsaverage6_hemi-{hemi}_desc-{roi}_mask.label.gii'),
                join(tpl_dir,
                     f'tpl-fsaverage6_hemi-{hemi}_desc-MMP_dseg.label.gii'))

    masks[roi] = mask.astype(bool)
    n_voxels = np.sum(mask)
    np.save(join(tpl_dir,
                 f'tpl-fsaverage6_hemi-{hemi}_desc-{roi}_mask.npy'), mask)
    print(f"Created {hemi} {roi} mask containing "
          f"{n_voxels} vertices")

# Create single parcellation map
mask_map = np.zeros(mmp.shape)
for i, mask_name in enumerate(masks):
    mask_map[masks[mask_name]] = i + 1

write_gifti(mask_map,
            join(tpl_dir,
                 f'tpl-fsaverage6_hemi-{hemi}_desc-ROIs_dseg.label.gii'),
            join(tpl_dir,
                 f'tpl-fsaverage6_hemi-{hemi}_desc-MMP_dseg.label.gii'))
