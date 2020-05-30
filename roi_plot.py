from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

task = 'black'
roi = 'EAC'

results_dir = '/jukebox/hasson/snastase/isc-confounds/results'

dfs = []
for hemi in ['L', 'R']:
    results = np.load(join(results_dir,
                           (f'results_task-{task}_roi-{roi}_'
                            f'hemi-{hemi}_desc-excl0_iscs.npy')),
                      allow_pickle=True).item()

    model_rename = {m: f'model {m}' for m in results}

    hemi_df = pd.DataFrame(results)
    hemi_df.rename(columns=model_rename, inplace=True)
    hemi_df = hemi_df.melt(var_name='confound model', value_name='ISC')
    hemi_df['hemisphere'] = hemi
    dfs.append(hemi_df)
df = pd.concat(dfs)

model0_L = df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'L')]
model0_R = df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'R')]

fig, ax = plt.subplots(figsize=(6, 10))
sns.barplot(x='ISC', y='confound model', hue='hemisphere',
            palette=sns.xkcd_palette(["windows blue", "amber"]),
            data=df, ax=ax)
ax.legend(loc='upper left', title='hemisphere')
ax.axvline(np.mean(model0_L), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["windows blue"])
ax.axvline(np.mean(model0_R), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["amber"])


# Let's try getting a single plot collapse across tasks
tasks = ['pieman', 'prettymouth', 'milkyway', 'slumlordreach',
         'notthefallintact', 'black', 'forgot']

dfs = []
for task in tasks:
    for hemi in ['L', 'R']:
        results = np.load(join(results_dir,
                               (f'results_task-{task}_roi-{roi}_'
                                f'hemi-{hemi}_desc-excl0_iscs.npy')),
                          allow_pickle=True).item()

        model_rename = {m: f'model {m}' for m in results}

        hemi_df = pd.DataFrame(results)
        hemi_df.rename(columns=model_rename, inplace=True)
        hemi_df = hemi_df.melt(var_name='confound model', value_name='ISC')
        hemi_df['hemisphere'] = hemi
        hemi_df['story'] = task
        dfs.append(hemi_df)
df = pd.concat(dfs)

model0_L = df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'L')]
model0_R = df['ISC'][(df['confound model'] == 'model 0') &
                     (df['hemisphere'] == 'R')]

fig, ax = plt.subplots(figsize=(6, 10))
sns.barplot(x='ISC', y='confound model', hue='hemisphere',
            palette=sns.xkcd_palette(["windows blue", "amber"]),
            data=df, ax=ax)
ax.legend(loc='upper left', title='hemisphere')
ax.axvline(np.mean(model0_L), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["windows blue"])
ax.axvline(np.mean(model0_R), zorder=0, linestyle='--',
           color=sns.xkcd_rgb["amber"])

#fig, ax = plt.subplots(figsize=(6, 10))
#sns.stripplot(x='ISC', y='confound model', hue='hemisphere',
#              palette=sns.xkcd_palette(["greyish", "greyish"]),
#              dodge=.532, data=df, ax=ax, zorder=0)
#sns.pointplot(x='ISC', y='confound model', hue='hemisphere',
#              palette=sns.xkcd_palette(["windows blue", "amber"]),
#              data=df, ax=ax, join=False, dodge=.532)
#ax.axvline(np.mean(model0_L), zorder=0, linestyle='--',
#           color=sns.xkcd_rgb["windows blue"])
#ax.axvline(np.mean(model0_R), zorder=0, linestyle='--',
#           color=sns.xkcd_rgb["amber"])