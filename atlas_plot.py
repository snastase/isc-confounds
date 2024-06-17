# conda activate confounds

from os.path import join
import json
from natsort import natsorted
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_rel
from statsmodels.stats.multitest import multipletests
from extract_confounds import load_confounds, extract_confounds
import nibabel as nib
from surfplot import Plot
from neuromaps.datasets import fetch_fslr
from neuromaps.transforms import fsaverage_to_fslr


space = 'fsaverage6'
parc = 'Schaefer1000'
hemis = ['L', 'R']

base_dir = '/jukebox/hasson/snastase/isc-confounds'
results_dir = '/jukebox/hasson/snastase/isc-confounds/results'
tpl_dir = join(base_dir, 'afni', f'tpl-{space}')

# Pick stimulus type, either narratives or movies
task_json = join(base_dir, 'narratives_meta.json')
#task_json = join(base_dir, 'movies_meta.json')

# Set up for multiple runs per task
tasks = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
         'notthefallintact', 'black', 'forgot']
task_runs = {t: [None] for t in tasks}

#task_runs = {'budapest': [1, 2, 3, 4, 5],
#             'life': [1, 2, 3, 4],
#             'raiders': [1, 2, 3, 4]}

# Get metadata for all subjects for a given task
with open(task_json) as f:
    task_meta = json.load(f)

# Get confound models
with open(join(base_dir, 'model_meta.json')) as f:
    model_meta = json.load(f)
models = model_meta.keys()
    
# Helper function for Fisher-transformed average
def fisher_mean(correlations, axis=None):
    return np.tanh(np.nanmean(np.arctanh(correlations), axis=axis))


# Function to expand parcel-level data back onto cortical map
def parcel_reconstruct(data, parc, n_vertices=40962,
                       exclude_zero=True, return_gii=False,
                       template_gii=None):
    labels = np.unique(parc).tolist()
    if 0 in labels and exclude_zero:
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

# Convience function to plot parcel maps
def plot_parcels(data_lh, data_rh, parc_lh, parc_rh,
                 color_range=None, cmap='YlOrRd', save_f=None): 
    data_lh = parcel_reconstruct(data_lh, parc_lh, return_gii=True,
                                 template_gii=join(tpl_dir,
                                    (f'tpl-fsaverage6_hemi-R_'
                                     f'desc-{parc}_dseg.label.gii')))
    data_rh = parcel_reconstruct(data_rh, parc_rh, return_gii=True,
                                 template_gii=join(tpl_dir,
                                    (f'tpl-fsaverage6_hemi-R_'
                                     f'desc-{parc}_dseg.label.gii')))
    
    data_lh = fsaverage_to_fslr(data_lh, target_density='32k',
                                hemi='L', method='linear')[0]
    data_rh = fsaverage_to_fslr(data_rh, target_density='32k',
                                hemi='R', method='linear')[0]
    
    surfaces_fslr = fetch_fslr()
    surf_lh, surf_rh = surfaces_fslr['inflated']
    
    p = Plot(surf_lh=surf_lh, 
             surf_rh=surf_rh,
             brightness=.7,
             zoom=1.58,
             size=(500, 375))
    p.add_layer({'left': data_lh,
                 'right': data_rh}, 
                color_range=color_range,
                cmap=cmap)
    fig = p.build(scale=(3, 3))
    if save_f:
        plt.savefig(save_f, dpi=300, transparent=True,
                    bbox_inches='tight')


# Load in atlas
parcs, parcs_gii = {}, {}
for hemi in ['L', 'R']:
    atlas_fn = join(tpl_dir,
                    (f'tpl-fsaverage6_hemi-{hemi}_'
                     f'desc-{parc}_dseg.label.gii'))
    parcs_gii[hemi] = nib.load(atlas_fn)
    parcs[hemi] = parcs_gii[hemi].agg_data()


# Load in parcellation maps
parc_maps = {}
for task in task_runs:
    parc_maps[task] = {}
    for run in task_runs[task]:
        if run:
            parc_maps[task][run] = {}
        else:
            parc_maps[task] = {}
        for hemi in ['L', 'R']:
            if run: 
                parc_fn = join(base_dir, 'results',
                    (f'results_task-{task}_run-{run:02d}_'
                     f'hemi-{hemi}_parc-{parc}_iscs.npy'))
                parc_maps[task][run][hemi] = np.load(
                    parc_fn, allow_pickle=True).item()
            else:
                parc_fn = join(base_dir, 'results',
                    (f'results_task-{task}_'
                     f'hemi-{hemi}_parc-{parc}_iscs.npy'))
                parc_maps[task][hemi] = np.load(
                    parc_fn, allow_pickle=True).item()


# Plot an example movie task, run, model
task, run, model = 'raiders', 4, '0'
parc_lh = fisher_mean(parc_maps[task][run]['L'][model], axis=0)
parc_rh = fisher_mean(parc_maps[task][run]['R'][model], axis=0)

plot_parcels(parc_lh, parc_rh, parcs['L'], parcs['R'],
             color_range=(0, .7), cmap='YlOrRd')

# Plot an example narrative task, run, model
task, model = 'pieman', '0'
parc_lh = fisher_mean(parc_maps[task]['L'][model], axis=0)
parc_rh = fisher_mean(parc_maps[task]['R'][model], axis=0)

plot_parcels(parc_lh, parc_rh, parcs['L'], parcs['R'],
             color_range=(0, .4), cmap='YlOrRd')


# Plot single ISC map across all movies for a single model
model = '0'
parc_all = {'L': [], 'R': []}
for hemi in ['L', 'R']:
    for task in task_runs:
        for run in task_runs[task]:
            parc_all[hemi].extend(parc_maps[task][run][hemi][model])
        
    parc_all[hemi] = np.mean(np.vstack(parc_all[hemi]), axis=0)

save_f = f'figures/isc_map_movies_model{model}.png'
plot_parcels(parc_all['L'], parc_all['R'], parcs['L'], parcs['R'],
             color_range=(0, .7), cmap='YlOrRd', save_f=save_f)


# Plot single ISC map across all narratives for a single model
model = '0'
parc_all = {'L': [], 'R': []}
for hemi in ['L', 'R']:
    for task in task_runs:
            parc_all[hemi].extend(parc_maps[task][hemi][model])
        
    parc_all[hemi] = np.mean(np.vstack(parc_all[hemi]), axis=0)

save_f = f'figures/isc_map_narratives_model{model}.png'
plot_parcels(parc_all['L'], parc_all['R'], parcs['L'], parcs['R'],
             color_range=(0, .3), cmap='YlOrRd', save_f=save_f)


# Extract subject-level (mean) framewise displacement values
exclude_json = join(base_dir, 'narratives_exclude.json')
#exclude_json = join(base_dir, 'movies_exclude.json')
with open(exclude_json) as f:
    task_exclude = json.load(f)
    
fd_df = {'stimulus': [], 'framewise displacement': []}

if task in ['budapest', 'life', 'raiders']:
    fd_df['run'] = []

for task in task_runs:
    for run in task_runs[task]:        
        for subject in task_meta[task]:
    
            # Check if we should a priori exclude
            if (task not in ['budapest', 'life', 'raiders']
                and subject in task_exclude[task]):
                continue
                
            if (task in task_exclude
                and subject in task_exclude[task]
                and run in task_exclude[task][subject]):
                continue
    
            # Grab confound files for multiple runs if present
            confound_fns = natsorted(task_meta[task][subject]['confounds'])

            if run:
                #confounds_fn = confound_fns[run - 1]
                confounds_fn = next(f for f in confound_fns
                                    if f'run-{run:02d}' in f)
                assert f'run-{run:02d}' in confounds_fn
            else:
                confounds_fn = confound_fns[0]
                
            confounds_df, confounds_meta = load_confounds(confounds_fn)
            
            fd = extract_confounds(confounds_df, confounds_meta,
                                   {'confounds': ['framewise_displacement']})
            fd = np.nanmean(fd['framewise_displacement'])
            
            fd_df['stimulus'].append(task)
            fd_df['framewise displacement'].append(fd)

            if run:
                fd_df['run'].append(run)
        
fd_df = pd.DataFrame(fd_df)


# Compute correlation with FD across all runs/tasks
parc_corr, rw_corr = {}, {}
for model in models:
    model_stack, fd_stack = [], []
    rw_corr[model] = {}
    for task in task_runs:
        rw_corr[model][task] = {}
        for run in task_runs[task]:
            rw_corr[model][task][run] = {}
            if run:
                parc_bh = np.hstack((
                    parc_maps[task][run]['L'][model],
                    parc_maps[task][run]['R'][model]))
            else:
                parc_bh = np.hstack((
                    parc_maps[task]['L'][model],
                    parc_maps[task]['R'][model]))

            # Be really careful that subjects (and runs) are in
            # the same sorted order from ISC analysis (roi_isc.py)

            if run:
                fd_run = fd_df[(fd_df['stimulus'] == task) &
                               (fd_df['run'] == run)][
                               'framewise displacement'].to_numpy()
            else:
                fd_run = fd_df[(fd_df['stimulus'] == task)][
                               'framewise displacement'].to_numpy()
            
            rs, ps = [], []
            for parc_ss in parc_bh.T:
                r, p = spearmanr(parc_ss, fd_run)
                rs.append(r)
                ps.append(p)

            if run: 
                rw_corr[model][task][run]['r'] = np.array(rs)
                rw_corr[model][task][run]['p'] = np.array(ps)
            else:
                rw_corr[model][task]['r'] = np.array(rs)
                rw_corr[model][task]['p'] = np.array(ps)
            
            model_stack.append(parc_bh)
            fd_stack.append(fd_run)
               
    model_stack = np.vstack(model_stack)
    fd_stack = np.concatenate(fd_stack)
    
    rs, ps = [], []
    for parc_ss in model_stack.T:
        r, p = spearmanr(parc_ss, fd_stack)
        rs.append(r)
        ps.append(p)

    _, qs, _, _ = multipletests(np.nan_to_num(ps, nan=1),
                                method='fdr_bh')
    
    parc_corr[model] = {'r': np.nan_to_num(rs),
                        'p': np.nan_to_num(ps, nan=1),
                        'q': qs}
    print(f"Finished ISC-FDR correlation for model {model}")


# Visualize ISC-FD correlation for a single example movie run
model, task, run = '0', 'raiders', 4

rw_ex = rw_corr[model][task][run]['r']
plot_parcels(rw_ex[:500], rw_ex[500:], parcs['L'], parcs['R'],
             cmap='coolwarm')

# Visualize ISC-FD correlation for a single example narratives run
model, task = '0', 'black'

rw_ex = rw_corr[model][task]['r']
plot_parcels(rw_ex[:500], rw_ex[500:], parcs['L'], parcs['R'],
             cmap='coolwarm')


# Visualize ISC-FD correlation across all movie runs
model = '20'

rs_ex = parc_corr[model]['r']
save_f = f'figures/isc-fd_map_movies_model{model}.png'
plot_parcels(rs_ex[:500], rs_ex[500:], parcs['L'], parcs['R'],
             cmap='coolwarm', color_range=(-.5, .5), save_f=save_f)

# Visualize ISC-FD correlation across all narrative runs
model = '20'

rs_ex = parc_corr[model]['r']
save_f = f'figures/isc-fd_map_narratives_model{model}.png'
plot_parcels(rs_ex[:500], rs_ex[500:], parcs['L'], parcs['R'],
             cmap='coolwarm', color_range=(-.3, .3), save_f=save_f)


# Visualize all parcel ISC-FDs in strip plot
n_parcels = 1000
corr_df = {'r': [], 'confound model': []}
for model in models:
    for p in np.arange(n_parcels):
        corr_df['r'].append(parc_corr[model]['r'][p])
        corr_df['confound model'].append(f'model {model}')
corr_df = pd.DataFrame(corr_df)
sns.set(font_scale=1.1, style='white')
fig, ax = plt.subplots(figsize=(6, 8))
sns.stripplot(x='r', y='confound model', color='.5', 
              alpha=.2, data=corr_df, ax=ax)
ax.axvline(0, zorder=0, linestyle='--', c='.7')


# Compute proportions and convert to DataFrame
n_parcels = 1000
thresh = .05
prop_df = {'proportion': [], 'confound model': []}
for model in models:
    prop = np.sum(parc_corr[model]['q'] < thresh) / n_parcels
    prop_df['confound model'].append(f'model {model}')
    prop_df['proportion'].append(prop)
prop_df = pd.DataFrame(prop_df)

model0 = prop_df[prop_df['confound model'] == 'model 0']['proportion']


# Plot without interrupted x-axis
sns.set(font_scale=1.3, style='ticks')
fig, ax = plt.subplots(figsize=(6.5, 6.5))
sns.barplot(x='proportion', y='confound model',
            color=sns.xkcd_rgb['grey'], data=prop_df, ax=ax)
ax.axvline(model0.to_numpy(), zorder=0, linestyle='--',
           c=sns.xkcd_rgb['grey'])
ax.set_xlabel((f'proportion of parcels with significant\n'
                f'ISC-FD correlation (FDR corrected)'))
ax.set_xlim(0, .8)
for m in np.arange(21):
    prop = prop_df[prop_df['confound model']
                   == f'model {m}']['proportion'].values[0]
    n_parc = prop * n_parcels
    if prop > .1:
        ax.annotate(f'{int(n_parc)}', xy=(prop - .004, m), c='w',
                    ha='right', va='center_baseline', size=11)
    else:
        ax.annotate(f'{int(n_parc)}', xy=(prop + .007, m),
                    c='.3', ha='left',
                    va='center_baseline', size=11)
sns.despine()
fig.savefig('figures/confounds_isc-fd_narratives_Schaefer1000.png', transparent=True, bbox_inches='tight', dpi=300)


# Plot with interrupted x-axis
sns.set(font_scale=1.3, style='ticks')
break_x = (.1, .6) # movies
fig, axs = plt.subplots(1, 2, figsize=(6.5, 6.5), sharey=True,
                        gridspec_kw={'wspace':0.03, 'width_ratios':[1,7]})
sns.barplot(x='proportion', y='confound model',
            color=sns.xkcd_rgb['grey'],
            estimator=fisher_mean, data=prop_df, ax=axs[0])
sns.barplot(x='proportion', y='confound model',
            color=sns.xkcd_rgb['grey'], data=prop_df, ax=axs[1])
axs[1].axvline(model0.to_numpy(), zorder=0, linestyle='--',
           c=sns.xkcd_rgb['grey'])
axs[1].set_xlabel((f'proportion of parcels with significant\n'
                   f'ISC-FD correlation (FDR corrected)'))
axs[1].xaxis.set_label_coords(0.408, -.08)
axs[0].set_xlim(0, break_x[0])
axs[1].set_xlim(left=break_x[1], right=.87)
axs[0].set_xticks([0])
axs[0].set_xticklabels([0])
axs[0].xaxis.label.set_visible(False)
axs[0].tick_params(axis='y', which='both', length=0)
axs[1].tick_params(axis='y', which='both', length=0)
for m in np.arange(21):
    prop = prop_df[prop_df['confound model']
                   == f'model {m}']['proportion'].values[0]
    n_parc = prop * n_parcels
    if prop > .1:
        axs[1].annotate(f'{int(n_parc)}', xy=(prop - .002, m), c='w',
                    ha='right', va='center_baseline', size=11)
    else:
        axs[1].annotate(f'{int(n_parc)}', xy=(prop + .002, m),
                    c='.3', ha='left',
                    va='center_baseline', size=11)
sns.despine(ax=axs[0])
sns.despine(ax=axs[1], left=True)
fig.savefig('figures/confounds_isc-fd_movies_Schaefer1000.png', transparent=True, bbox_inches='tight', dpi=300)
