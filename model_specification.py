from os.path import join
from natsort import natsorted
import json

base_dir = '/jukebox/hasson/snastase/isc-confounds'


# Create a dictionary for confound regression models
model_meta = {'0': {'confounds': []},
              '1': {'confounds':
                    ['trans_x', 'trans_y', 'trans_z',
                     'rot_x', 'rot_y', 'rot_z']},
              '2': {'confounds':
                    ['trans_x', 'trans_y', 'trans_z',
                     'rot_x', 'rot_y', 'rot_z',
                     'trans_x_derivative1',
                     'trans_y_derivative1',
                     'trans_z_derivative1',
                     'rot_x_derivative1',
                     'rot_y_derivative1',
                     'rot_z_derivative1']},
              '3': {'confounds':
                    ['trans_x', 'trans_y', 'trans_z',
                     'rot_x', 'rot_y', 'rot_z',
                     'trans_x_derivative1',
                     'trans_y_derivative1',
                     'trans_z_derivative1',
                     'rot_x_derivative1',
                     'rot_y_derivative1',
                     'rot_z_derivative1',
                     'trans_x_power2', 'trans_y_power2',
                     'trans_z_power2', 'rot_x_power2',
                     'rot_y_power2', 'rot_z_power2',
                     'trans_x_derivative1_power2',
                     'trans_y_derivative1_power2',
                     'trans_z_derivative1_power2',
                     'rot_x_derivative1_power2',
                     'rot_y_derivative1_power2',
                     'rot_z_derivative1_power2']},
              'X': {'confounds':
                    ['trans_x', 'trans_y', 'trans_z',
                     'rot_x', 'rot_y', 'rot_z',
                     'trans_x_derivative1',
                     'trans_y_derivative1',
                     'trans_z_derivative1',
                     'rot_x_derivative1',
                     'rot_y_derivative1',
                     'rot_z_derivative1',
                     'framewise_displacement',
                     'cosine'],
                    'aCompCor':
                    [{'n_comps': 5, 'tissue': 'CSF'},
                     {'n_comps': 5, 'tissue': 'WM'}]}}
        
with open(join(base_dir, 'model_meta.json'), 'w') as f:
    json.dump(model_meta, f, indent=2)