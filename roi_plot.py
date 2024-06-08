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


roi = 'MT+' # EAC, IFG, V1, MT+
hemis = ['L', 'R']

base_dir = '/jukebox/hasson/snastase/isc-confounds'
results_dir = '/jukebox/hasson/snastase/isc-confounds/results'

# Pick stimulus type, either narratives or movies
#task_json = join(base_dir, 'narratives_meta.json')
task_json = join(base_dir, 'movies_meta.json')

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


# Let's try getting a single plot collapse across tasks
# Set up for multiple runs per task
#tasks = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
#         'notthefallintact', 'black', 'forgot']
#task_runs = {t: [None] for t in tasks}

task_runs = {'budapest': [1, 2, 3, 4, 5],
             'life': [1, 2, 3, 4],
             'raiders': [1, 2, 3, 4]}
dfs = []
for task in task_runs:
    for run in task_runs[task]:
        for hemi in ['L', 'R']:
    
            if run:
                results = np.load(join(results_dir,
                                       (f'results_task-{task}_run-{run:02d}_'
                                        f'hemi-{hemi}_roi-{roi}_'
                                        f'desc-excl0_iscs.npy')),
                                  allow_pickle=True).item()
            else:
                results = np.load(join(results_dir,
                                       (f'results_task-{task}_'
                                        f'hemi-{hemi}_roi-{roi}_'
                                        f'desc-excl0_iscs.npy')),
                                  allow_pickle=True).item()
    
            model_rename = {m: f'model {m}' for m in results}
    
            hemi_df = pd.DataFrame(results)
            hemi_df.rename(columns=model_rename, inplace=True)
            hemi_df = hemi_df.melt(var_name='confound model', value_name='ISC')
            hemi_df['hemisphere'] = hemi
            hemi_df['stimulus'] = task

            if run:
                hemi_df['run'] = run
            
            dfs.append(hemi_df)
df = pd.concat(dfs, ignore_index=True)


model0_L = df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'L')]
model0_R = df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'R')]

sns.set(font_scale=1.1, style='white')
fig, ax = plt.subplots(figsize=(6, 8))
sns.barplot(x='ISC', y='confound model', hue='hemisphere',
            data=df, ax=ax, estimator=fisher_mean,
            palette=sns.xkcd_palette(["windows blue", "amber"]))
ax.legend(loc='lower left', title='hemisphere')
ax.axvline(fisher_mean(model0_L), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["windows blue"])
ax.axvline(fisher_mean(model0_R), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["amber"])
ax.set_xlabel(f'ISC ({roi})')
#ax.set_xlim(0, .55) # EAC
#ax.set_xlim(0, .3) # IFG
sns.despine()
plt.savefig('figures/confounds_isc_movies_MT+.svg', transparent=True,
            bbox_inches='tight', dpi=300)


# T-tests against model 0
ttest = {}
model0 = fisher_mean([df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'L')],
                     df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'R')]], axis=0)
for model in list(models)[1:]:
    compare = fisher_mean([df['ISC'][(df['confound model'] == f'model {model}') &
                          (df['hemisphere'] == 'L')],
                          df['ISC'][(df['confound model'] == f'model {model}') &
                          (df['hemisphere'] == 'R')]], axis=0)
    t, p = ttest_rel(np.arctanh(compare), np.arctanh(model0))
    ttest[model] = {'t': t, 'p': p}

ps = [ttest[m]['p'] for m in ttest.keys()]
_, qs, _, _ = multipletests(ps, method='fdr_bh')

for i, model in enumerate(ttest.keys()):
    ttest[model]['q'] = qs[i]
    

# Extract subject-level (mean) framewise displacement values
#exclude_json = join(base_dir, 'narratives_exclude.json')
exclude_json = join(base_dir, 'movies_exclude.json')
with open(exclude_json) as f:
    task_exclude = json.load(f)
    
fd_df = {'story': [], 'framewise displacement': []}

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
            
            fd_df['story'].append(task)
            fd_df['framewise displacement'].append(fd)

            if run:
                fd_df['run'].append(run)
        
fd_df = pd.DataFrame(fd_df)


# Correlate ISCs for each model with framewise displacement
corr_df = {'hemisphere': [], 'confound model': [], 'r': [], 'p': []}
for hemi in hemis:
    for model in np.arange(21):
        r, p = spearmanr(
            df['ISC'][
                (df['confound model'] == f'model {model}') &
                (df['hemisphere'] == hemi)],
            fd_df['framewise displacement'])
        corr_df['hemisphere'].append(hemi)
        corr_df['confound model'].append(f'model {model}')
        corr_df['r'].append(r)
        corr_df['p'].append(p)

corr_df = pd.DataFrame(corr_df)

_, q, _, _ = multipletests(corr_df['p'], method='fdr_bh')
corr_df['FDR q'] = q

model0_L = corr_df['r'][(corr_df['confound model'] == 'model 0') &
                        (corr_df['hemisphere'] == 'L')]
model0_R = corr_df['r'][(corr_df['confound model'] == 'model 0') &
                        (corr_df['hemisphere'] == 'R')]

sns.set(font_scale=1.1, style='white')
fig, ax = plt.subplots(figsize=(6, 8))
sns.barplot(x='r', y='confound model', hue='hemisphere',
            palette=sns.xkcd_palette(["windows blue", "amber"]),
            estimator=fisher_mean, data=corr_df, ax=ax)
ax.legend(loc='lower left', title='hemisphere')
ax.axvline(fisher_mean(model0_L), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["windows blue"])
ax.axvline(fisher_mean(model0_R), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["amber"])
ax.set_xlabel(f'Spearman r between ISC and FD (V1)')
#ax.set_xlim(-.28, 0)
sns.despine(left=True, right=False)
fig.savefig('figures/confounds_isc-fd_movies_MT+.svg', transparent=True, bbox_inches='tight', dpi=300)


# Scatterplot
model = 'model 0'
mean_r = fisher_mean(corr_df['r'][
    corr_df['confound model'] == model])

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(fd_df['framewise displacement'],
           df['ISC'][(df['confound model'] == model) &
                     (df['hemisphere'] == 'R')],
           c=sns.xkcd_rgb["amber"])
ax.scatter(fd_df['framewise displacement'],
           df['ISC'][(df['confound model'] == model) &
                     (df['hemisphere'] == 'L')],
           c=sns.xkcd_rgb["windows blue"])
#ax.set_xlim(0.05, .35)
#ax.set_ylim(-.25, .85)
ax.set_xlabel('framewise displacement')
ax.set_ylabel('ISC')
ax.annotate(f'r = {mean_r:.3f}', ha='right',
            xy=(.95, .05), xycoords='axes fraction')
ax.set_title(model)
sns.despine()
plt.savefig('scatter_model13.svg', transparency=True, bbox_to_inches='tight', dpi=300)