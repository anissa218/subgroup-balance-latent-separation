import os
import yaml
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
import matplotlib.pyplot as plt
from scipy.stats import linregress, t
from collections import defaultdict
import seaborn as sns

def get_metrics_one_run(preds_metadata_df, attr, experiment, prop, run_id, loss = torch.nn.CrossEntropyLoss()):
    acc = accuracy_score(preds_metadata_df['target'], preds_metadata_df['prediction'])
    balanced_acc = balanced_accuracy_score(preds_metadata_df['target'], preds_metadata_df['prediction'])

    logits = np.stack([preds_metadata_df['logits_0'], preds_metadata_df['logits_1']], axis=1)
    probs = softmax(logits, axis=1)
    preds_metadata_df['pos_prob'] = probs[:, 1]
    auc = roc_auc_score(preds_metadata_df['target'], preds_metadata_df['pos_prob'])

    group_0 = preds_metadata_df[preds_metadata_df[attr] == 0]
    group_1 = preds_metadata_df[preds_metadata_df[attr] == 1]

    acc_group_0 = accuracy_score(group_0['target'], group_0['prediction']) if not group_0.empty else None
    acc_group_1 = accuracy_score(group_1['target'], group_1['prediction']) if not group_1.empty else None

    balanced_acc_group_0 = balanced_accuracy_score(group_0['target'], group_0['prediction']) if not group_0.empty else None
    balanced_acc_group_1 = balanced_accuracy_score(group_1['target'], group_1['prediction']) if not group_1.empty else None

    logits_0 = np.stack([group_0['logits_0'], group_0['logits_1']], axis=1)
    losses_group_0 = (loss(torch.tensor(logits_0, dtype=torch.float32), torch.tensor(group_0['target'].values, dtype=torch.long)).item())
    auc_group_0 = roc_auc_score(group_0['target'], group_0['pos_prob'])

    logits_1 = np.stack([group_1['logits_0'], group_1['logits_1']], axis=1)
    losses_group_1 = (loss(torch.tensor(logits_1, dtype=torch.float32), torch.tensor(group_1['target'].values, dtype=torch.long)).item())
    auc_group_1 = roc_auc_score(group_1['target'], group_1['pos_prob'])
                
    run_record = {
                    'experiment': experiment,
                    'subgroup': attr,
                    'proportion': prop,
                    'run_id': run_id,
                    'acc_group_0': acc_group_0,
                    'acc_group_1': acc_group_1,
                    'acc_mean': acc,
                    'balanced_acc_group_0': balanced_acc_group_0,
                    'balanced_acc_group_1': balanced_acc_group_1,
                    'balanced_acc_mean': balanced_acc,
                    'auc_mean': auc,
                    'auc_group_0': auc_group_0,
                    'auc_group_1': auc_group_1,
                    'loss_group_0': losses_group_0,
                    'loss_group_1': losses_group_1,
                }
    return run_record

def get_summary_df(experiment_string, test_metadata_df, required_lr, required_data_name, attributes, proportions = range(0, 101, 10), max_count = 9):
    all_run_records = []
    for attr in attributes:
        for prop in proportions:
            experiment = experiment_string+'_'+attr+'_'+str(prop)

            results_dir = os.path.join('logs', experiment, 'runs')

            if not os.path.isdir(results_dir):
                continue

            count = 0

            for run_folder in os.listdir(results_dir):

                if count >= max_count:# some unnecessary extra seeds ran  
                    break

                with open(os.path.join(results_dir, run_folder,'.hydra','config.yaml'), "r") as file:
                    config = yaml.safe_load(file)

                if all(x in config['data']['train_data_path'] for x in [attr,required_data_name]) and config['model']['optimizer']['lr'] == required_lr: # filter out runs not for correct target, modified proportion, and not using the full dataset

                    preds_file = os.path.join(results_dir, run_folder, 'preds.csv')

                    if not os.path.isfile(preds_file):
                        print('Skipping:', preds_file)
                        continue
                        
                    preds_df = pd.read_csv(preds_file)
                    preds_metadata_df = pd.merge(preds_df, test_metadata_df, on='id', how='left')

                    run_record = get_metrics_one_run(preds_metadata_df,attr, experiment,prop,count)
                    all_run_records.append(run_record)

                    count += 1
        
    all_runs_df = pd.DataFrame(all_run_records)

    summary_df = all_runs_df.groupby('experiment').agg({
            'balanced_acc_group_0': ['mean', 'std'],
            'balanced_acc_group_1': ['mean', 'std'],
            'balanced_acc_mean': ['mean', 'std'],
            'acc_group_0': ['mean', 'std'],
            'acc_group_1': ['mean', 'std'],
            'acc_mean': ['mean', 'std'],
            'loss_group_0': ['mean', 'std'],
            'loss_group_1': ['mean', 'std'],
            'auc_mean': ['mean', 'std', 'count'],
            'auc_group_0': ['mean', 'std'],
            'auc_group_1': ['mean', 'std'],
        })
    summary_df['experiment'] = summary_df.index
    summary_df['subgroup'] = summary_df['experiment'].str.extract(r'alloc_(.+)_\d+$')
    summary_df['proportion'] = summary_df['experiment'].str.split('_').str[-1].astype(int)
    summary_df['count'] = summary_df[('auc_mean', 'count')]
    summary_df.drop(columns=[('auc_mean', 'count')], inplace=True)

    return all_runs_df, summary_df

def get_baseline_subgroup_df(base_dir,test_metadata_df,attributes,required_lr):

    scores_by_group = defaultdict(lambda: defaultdict(list))  # scores_by_group[attr][group_val] = [scores]

    for run in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run)
        if not os.path.exists(os.path.join(run_path, 'features.pt')):
            continue

        with open(os.path.join(run_path, '.hydra', 'config.yaml'), "r") as file:
            config = yaml.safe_load(file)

        if not config['model']['optimizer']['lr'] == required_lr:
            continue

        test_preds = pd.read_csv(os.path.join(run_path, 'preds.csv'))
        preds_metadata_df = pd.merge(test_preds, test_metadata_df, on='id', how='left')

        for attribute in attributes:
            for group_val in preds_metadata_df[attribute].dropna().unique():
                group_df = preds_metadata_df[preds_metadata_df[attribute] == group_val]
                if len(group_df) == 0:
                    continue

                acc = balanced_accuracy_score(group_df['target'], group_df['prediction'])
                scores_by_group[attribute][group_val].append(acc)
        
        summary_rows = []

    for attr, group_scores in scores_by_group.items():
        for group_val, scores in group_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores, ddof=1)
            summary_rows.append({
                'attribute': attr,
                'group': group_val,
                'mean_balanced_accuracy': mean_score,
                'std_balanced_accuracy': std_score
            })

    summary_df = pd.DataFrame(summary_rows)

    summary_gap_df = pd.DataFrame()
    summary_gap_df['subgroup'] = summary_df['attribute'].unique()

    mean_diffs = []
    std_diffs = []
    mean_1s = []
    std_1s = []
    mean_0s = []
    std_0s = []

    for subgroup in summary_gap_df['subgroup']:
        mean_1 = summary_df.loc[
            (summary_df['attribute'] == subgroup) & (summary_df['group'] == 1.0),
            'mean_balanced_accuracy'
        ].values[0]
        std_1 = summary_df.loc[
            (summary_df['attribute'] == subgroup) & (summary_df['group'] == 1.0),
            'std_balanced_accuracy'
        ].values[0]

        mean_0 = summary_df.loc[
            (summary_df['attribute'] == subgroup) & (summary_df['group'] == 0.0),
            'mean_balanced_accuracy'
        ].values[0]
        std_0 = summary_df.loc[
            (summary_df['attribute'] == subgroup) & (summary_df['group'] == 0.0),
            'std_balanced_accuracy'
        ].values[0]

        mean_diffs.append(mean_1 - mean_0)
        std_diffs.append(np.sqrt(std_1**2 + std_0**2))
        mean_1s.append(mean_1)
        std_1s.append(std_1)
        mean_0s.append(mean_0)
        std_0s.append(std_0)

    # add everything into df
    summary_gap_df['mean_balanced_acc_group_1'] = mean_1s
    summary_gap_df['std_balanced_acc_group_1'] = std_1s
    summary_gap_df['mean_balanced_acc_group_0'] = mean_0s
    summary_gap_df['std_balanced_acc_group_0'] = std_0s
    summary_gap_df['mean_balanced_acc_diff'] = np.abs(mean_diffs)
    summary_gap_df['std_balanced_acc_diff'] = std_diffs

    return summary_gap_df
                
def plot_slope(filtered_df, summary_df, metric, min_val, max_val, dataset,title='',save_path=''):
    for subgroup in filtered_df['subgroup'].unique():

        # first bit: just plot average point
        subgroup_df = summary_df[summary_df['subgroup'] == subgroup]
        subgroup_df.sort_values(by='proportion', inplace=True)
        x = subgroup_df['proportion']/100
        y0 = subgroup_df[(metric+'_group_0','mean')][::-1]
        y1 = subgroup_df[(metric+'_group_1','mean')]

        if subgroup == 'PerformedProcedureStepDescription_binary':
            plt.scatter(x, y0, label='Portable', marker='o', color='#1f77b4')
            plt.scatter(x, y1, label='Fixed', marker='o', color='darkorange')
        elif subgroup == 'Gender_binary' and dataset=='MIMIC':
            plt.scatter(x, y0, label='Female', marker='o', color='#1f77b4')
            plt.scatter(x, y1, label='Male', marker='o', color='darkorange')
        elif subgroup == 'Gender_binary':
            plt.scatter(x, y0, label='No mention', marker='o', color='#1f77b4')
            plt.scatter(x, y1, label='Mentioned', marker='o', color='darkorange')
        elif subgroup == 'Dataset_binary':
            plt.scatter(x, y0, label='Source A', marker='o', color='#1f77b4')
            plt.scatter(x, y1, label='Source B', marker='o', color='darkorange')
        else:
            plt.scatter(x, y0, label='Group 0', marker='o', color='#1f77b4')
            plt.scatter(x, y1, label='Group 1', marker='o', color='darkorange')

        # # then do regression: 
        subgroup_df = filtered_df[filtered_df['subgroup']==subgroup]
        x = subgroup_df['proportion']/100
        y_0 = subgroup_df[metric+'_group_0'].values[::-1]
        y_1 = subgroup_df[metric+'_group_1']

        res0 = linregress(x, y_0)
        b_0, a_0, std_b0 = res0.slope, res0.intercept, res0.stderr

        res1 = linregress(x, y_1)
        b_1, a_1, std_b1 = res1.slope, res1.intercept, res1.stderr

        interval = np.linspace(0, 1, 100)

        y0_fit = a_0 + b_0 * interval
        y1_fit = a_1 + b_1 * interval
        y0_lower = a_0 + (b_0 - std_b0) * interval
        y0_upper = a_0 + (b_0 + std_b0) * interval
        y1_lower = a_1 + (b_1 - std_b1) * interval
        y1_upper = a_1 + (b_1 + std_b1) * interval

        plt.plot(interval, y0_fit, color='#1f77b4')
        plt.plot(interval, y1_fit, color='darkorange')
        plt.fill_between(interval, y0_lower, y0_upper, color='#1f77b4', alpha=0.2)
        plt.fill_between(interval, y1_lower, y1_upper, color='darkorange', alpha=0.2)

        ax = plt.gca()

        # choose a fixed x to place labels (0..1 because interval is 0..1)
        x_label_pos = 0.95

        # compute y from fit and clamp into visible y-range with a tiny margin
        margin = 0.01 * (max_val - min_val)

        y0_pos = a_0 + b_0 * x_label_pos
        y1_pos = a_1 + b_1 * x_label_pos

        y0_pos = np.clip(y0_pos, min_val + margin, max_val - margin)
        y1_pos = np.clip(y1_pos, min_val + margin, max_val - margin)

        # optional small vertical nudge so the two labels don't overlap
        if abs(y0_pos - y1_pos) < 1.5 * margin:
            y1_pos += 1.5 * margin

        ax.annotate(f"a = {b_0:.3f}", xy=(x_label_pos, y0_pos), xycoords='data',
                    ha='right', va='center', fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                    color='#1f77b4')

        ax.annotate(f"a = {b_1:.3f}", xy=(x_label_pos, y1_pos), xycoords='data',
                    ha='right', va='center', fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8),
                    color='darkorange')

        plt.xlabel('Proportion of subgroup in training data')
        if metric == 'balanced_acc':
            plt.ylabel('Subgroup accuracy')
        else:
            plt.ylabel('Subgroup ' + metric)
        plt.ylim(min_val, max_val)
        # plt.ylim(0.64,0.81)
        if title:
            plt.title(title)
        else:
            if subgroup=='PerformedProcedureStepDescription_binary':
                plt.title(dataset+ ' Procedure')
            else:
                plt.title(dataset+' '+subgroup.removesuffix("_binary"))
        plt.legend()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

def get_slopes_df(filtered_df, metrics, subgroups, max_count=9):
    rows = []

    for subgroup in subgroups:
        subgroup_df = filtered_df[filtered_df['subgroup'] == subgroup]
        x = subgroup_df['proportion'] / 100  # normalize to 0–1

        row = {'subgroup': subgroup}

        for metric in metrics:
            y_0 = subgroup_df[f'{metric}_group_0'].values[::-1] # reversed 
            y_1 = subgroup_df[f'{metric}_group_1']
            
            # group 0
            res0 = linregress(x, y_0)
            b_0, a_0, std_b0 = res0.slope, res0.intercept, res0.stderr

            # group 1
            res1 = linregress(x, y_1)
            b_1, a_1, std_b1 = res1.slope, res1.intercept, res1.stderr

            mean_b = (b_0 + b_1) / 2
            mean_std = (std_b0 + std_b1) / 2  # rough average, you could also propagate properly

            row[f'{metric}_slope_group_0'] = b_0
            row[f'{metric}_slope_group_1'] = b_1
            row[f'{metric}_slope_mean'] = mean_b
            row[f'{metric}_slope_group_0_std'] = std_b0 / np.sqrt(max_count)
            row[f'{metric}_slope_group_1_std'] = std_b1 / np.sqrt(max_count)
            row[f'{metric}_slope_std'] = mean_std / np.sqrt(max_count)

            # also add information on performance change
            p0_exists = (subgroup_df['proportion'] == 0).any()
            p100_exists = (subgroup_df['proportion'] == 100).any()

            if not (p0_exists and p100_exists):
                row[f'{metric}_diff_group_0'] = np.nan
                row[f'{metric}_diff_group_1'] = np.nan
                row[f'{metric}_diff_mean'] = np.nan
                row[f'{metric}_diff_group_0_std'] = np.nan
                row[f'{metric}_diff_group_1_std'] = np.nan
                row[f'{metric}_diff_std'] = np.nan
            
            else:
                zero_shot_0_vals = subgroup_df[subgroup_df['proportion']==100][metric+'_group_0'].values
                full_shot_0_vals = subgroup_df[subgroup_df['proportion']==0][metric+'_group_0'].values  # because group 0 and 1 are flipped
                zero_shot_1_vals = subgroup_df[subgroup_df['proportion']==0][metric+'_group_1'].values
                full_shot_1_vals = subgroup_df[subgroup_df['proportion']==100][metric+'_group_1'].values

                diff_0 = full_shot_0_vals - zero_shot_0_vals
                diff_1 = full_shot_1_vals - zero_shot_1_vals

                row[f'{metric}_diff_group_0'] = np.mean(diff_0)
                row[f'{metric}_diff_group_1'] = np.mean(diff_1)
                row[f'{metric}_diff_mean'] = (np.mean(diff_0) + np.mean(diff_1)) / 2

                # add stds (or use sem if you want error bars)
                row[f'{metric}_diff_group_0_std'] = np.std(diff_0, ddof=1)/np.sqrt(max_count) if len(diff_0) > 1 else np.nan
                row[f'{metric}_diff_group_1_std'] = np.std(diff_1, ddof=1)/np.sqrt(max_count) if len(diff_1) > 1 else np.nan
                row[f'{metric}_diff_std'] = (row[f'{metric}_diff_group_0_std'] + row[f'{metric}_diff_group_1_std']) / 2

        rows.append(row)

    slopes_df = pd.DataFrame(rows)
    slopes_df['subgroup'] = slopes_df['subgroup'].str.replace("_binary", "", regex=False)   

    return slopes_df    

def plot_distance_slope(slopes_df,distance_metrics, metric,save_fig=False,save_path = 'final_figs/distances_slope.pdf'):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for ax, distance_metric in zip(axes, distance_metrics):
        x = slopes_df[f'{distance_metric}_g0_vs_g1_mean'].values
        xerr = slopes_df[f'{distance_metric}_g0_vs_g1_std'].values

        y = slopes_df[f'{metric}_slope_mean'].values
        yerr = slopes_df[f'{metric}_slope_std'].values

        # scatter with error bars
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt='o', alpha=0.5, ecolor='skyblue', capsize=3, markersize=5
        )
        
        # regression fit
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # smooth grid for line + CI
        x_fit = np.linspace(min(x), max(x), 200)
        y_fit = slope * x_fit + intercept
        if p_value < 0.01:
            p_text = "p<0.01"
        elif p_value < 0.05:
            p_text = "p<0.05"
        else:
            p_text = "n.s."  # not significant

        # compute confidence interval for regression line
        n = len(x)
        dof = n - 2  # degrees of freedom
        tval = t.ppf(0.975, dof)  # 95% CI
        
        # residuals
        y_pred = slope * x + intercept
        residuals = y - y_pred
        s_err = np.sqrt(np.sum(residuals**2) / dof)

        # standard error of predicted y values
        conf = tval * s_err * np.sqrt(1/n + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        
        # plot regression line + CI shading
        ax.plot(x_fit, y_fit, color='#1f77b4', label=f"r = {r_value**2:.2f}, {p_text}")
        ax.fill_between(x_fit, y_fit - conf, y_fit + conf, color='#1f77b4', alpha=0.1)

        # axis labels etc.
        ax.set_xlabel(f'Latent representation {distance_metric} between 2 subgroups')
        ax.set_ylabel(f'Mean slope of {metric} across subgroup allocations')
        # ax.set_title(distance_metric)
        ax.set_xlim(min(x)-0.02*max(x), 1.02*max(x))
        ax.legend()
        print(p_value)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

def plot_distance_generalisation(slopes_df,distance_metrics,metric,save_fig=False,save_path='final_figs/distance_loss_generalisation.pdf'):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, distance_metric in zip(axes, distance_metrics):
        x = slopes_df[f'{distance_metric}_g0_vs_g1_mean'].values
        xerr = slopes_df[f'{distance_metric}_g0_vs_g1_std'].values

        y = slopes_df[f'{metric}_diff_mean'].values
        yerr = slopes_df[f'{metric}_diff_std'].values

        # scatter with error bars
        ax.errorbar(
            x, y,
            xerr=xerr, yerr=yerr,
            fmt='o', alpha=0.5, ecolor='skyblue', capsize=3, markersize=5
        )
        
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        x_fit = np.linspace(min(x), max(x), 200)
        y_fit = slope * x_fit + intercept
        n = len(x)
        dof = n - 2  # degrees of freedom
        tval = t.ppf(0.975, dof)  # 95% CI
        y_pred = slope * x + intercept
        residuals = y - y_pred
        s_err = np.sqrt(np.sum(residuals**2) / dof)
        conf = tval * s_err * np.sqrt(1/n + (x_fit - np.mean(x))**2 / np.sum((x - np.mean(x))**2))
        
        if p_value < 0.01:
            p_text = "p<0.01"
        elif p_value < 0.05:
            p_text = "p<0.05"
        else:
            p_text = "n.s."  # not significant

        ax.plot(x_fit, y_fit, color='#1f77b4', label=f"r = {r_value**2:.2f}, {p_text}")
        ax.fill_between(x_fit, y_fit - conf, y_fit + conf, color='#1f77b4', alpha=0.1)
        
        ax.set_xlabel(f'Latent representation {distance_metric} between 2 subgroups')
        ax.set_ylabel(f'Full shot {metric} - zero shot')
        ax.set_title(distance_metric)
        ax.set_xlim(min(x)-0.02*max(x), 1.02*max(x))

        ax.legend()
        print(p_value)

    plt.tight_layout()

    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()


def plot_metric_bars(summary_perf_df, metric, proportions=[0, 50, 100],ylim=[0,1.0],save_fig=False,save_path='final_figs/allocation_gains.pdf'):
    """
    Bar plot of mean performance at proportions 0, 50, 100 for each subgroup,
    with error bars showing std.
    
    summary_perf_df: DataFrame with columns like '{metric}_mean_at_{prop}' and '{metric}_std_at_{prop}'
    metric: str, one of ['auc', 'balanced_acc', 'loss', 'acc']
    proportions: list of proportions to plot (default: [0, 50, 100])
    """
    palette = sns.color_palette("colorblind", 10)

    fig, ax = plt.subplots(figsize=(8, 5))

    labels = summary_perf_df['subgroup'].tolist()
    x = np.arange(len(labels))  # subgroup positions
    width = 0.25  # bar width

    for i, prop in enumerate(proportions):
        means = summary_perf_df[f"{metric}_mean_at_{prop}"].values
        stds = summary_perf_df[f"{metric}_std_at_{prop}"].values
        ax.bar(
            x + i * width, means, width,
            yerr=stds, capsize=5,
            label=f"{prop}%",
            alpha=1,
            color=palette[i]
        )

    if metric=='balanced_acc':
        ax.set_ylabel(f"Mean subgroup balanced accuracy")
    else:
        ax.set_ylabel(f"Mean subgroup {metric}")

    ax.set_xlabel("")
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=0, ha='center')
    ax.legend(title='Allocation')

    plt.tight_layout()
    plt.ylim(ylim)

    if save_fig:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()

import os
import yaml
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from scipy.special import softmax

def safe_auc(y_true, y_score):
    # returns np.nan if AUC cannot be computed (e.g., only one class present)
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan

def safe_loss_for_group(group_df, loss_fn):
    if group_df.empty:
        return np.nan
    logits = np.stack([group_df['logits_0'], group_df['logits_1']], axis=1)
    with torch.no_grad():
        l = loss_fn(torch.tensor(logits, dtype=torch.float32),
                    torch.tensor(group_df['target'].values, dtype=torch.long)).item()
    return l

def get_metrics_one_run_all_attrs(preds_metadata_df, attrs_to_eval, experiment, prop, run_id,
                                  loss_fn=torch.nn.CrossEntropyLoss()):
    # compute global measures
    acc = accuracy_score(preds_metadata_df['target'], preds_metadata_df['prediction'])
    balanced_acc = balanced_accuracy_score(preds_metadata_df['target'], preds_metadata_df['prediction'])

    logits_full = np.stack([preds_metadata_df['logits_0'], preds_metadata_df['logits_1']], axis=1)
    probs = softmax(logits_full, axis=1)
    preds_metadata_df = preds_metadata_df.copy()
    preds_metadata_df['pos_prob'] = probs[:, 1]
    auc = safe_auc(preds_metadata_df['target'], preds_metadata_df['pos_prob'])

    record = {
        'experiment': experiment,
        'proportion': prop,
        'run_id': run_id,
        'acc_mean': acc,
        'balanced_acc_mean': balanced_acc,
        'auc_mean': auc,
    }

    # for each attribute, compute metrics for group 0 and group 1
    for attr in attrs_to_eval:
        group0 = preds_metadata_df[preds_metadata_df[attr] == 0]
        group1 = preds_metadata_df[preds_metadata_df[attr] == 1]

        # accuracy
        record[f'acc_{attr}_0'] = accuracy_score(group0['target'], group0['prediction']) if not group0.empty else np.nan
        record[f'acc_{attr}_1'] = accuracy_score(group1['target'], group1['prediction']) if not group1.empty else np.nan

        # balanced accuracy
        record[f'balanced_acc_{attr}_0'] = balanced_accuracy_score(group0['target'], group0['prediction']) if not group0.empty else np.nan
        record[f'balanced_acc_{attr}_1'] = balanced_accuracy_score(group1['target'], group1['prediction']) if not group1.empty else np.nan

        # auc
        record[f'auc_{attr}_0'] = safe_auc(group0['target'], group0['pos_prob']) if not group0.empty else np.nan
        record[f'auc_{attr}_1'] = safe_auc(group1['target'], group1['pos_prob']) if not group1.empty else np.nan

        # loss
        record[f'loss_{attr}_0'] = safe_loss_for_group(group0, loss_fn)
        record[f'loss_{attr}_1'] = safe_loss_for_group(group1, loss_fn)

    return record

def get_summary_df_all_attrs(experiment_string, test_metadata_df, required_lr, required_data_name,
                             attributes, proportions=range(0, 101, 10), max_count=9,
                             loss_fn=torch.nn.CrossEntropyLoss()):
    all_run_records = []
    for attr in attributes:
        for prop in proportions:
            experiment = experiment_string + '_' + attr + '_' + str(prop)
            results_dir = os.path.join('logs', experiment, 'runs')
            if not os.path.isdir(results_dir):
                continue

            count = 0
            for run_folder in os.listdir(results_dir):
                if count >= max_count:
                    break

                config_path = os.path.join(results_dir, run_folder, '.hydra', 'config.yaml')
                if not os.path.isfile(config_path):
                    continue

                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)

                # same filtering you used: check train path contains attr and dataset name, and lr matches
                if (all(x in config['data']['train_data_path'] for x in [attr, required_data_name])
                        and config['model']['optimizer']['lr'] == required_lr):

                    preds_file = os.path.join(results_dir, run_folder, 'preds.csv')
                    if not os.path.isfile(preds_file):
                        print('Skipping:', preds_file)
                        continue

                    preds_df = pd.read_csv(preds_file)
                    preds_metadata_df = pd.merge(preds_df, test_metadata_df, on='id', how='left')

                    run_record = get_metrics_one_run_all_attrs(preds_metadata_df, attributes, experiment, prop, count, loss_fn)
                    # also include which attribute was varied (so you can still see subgroup that was varied)
                    run_record['varied_attr'] = attr
                    all_run_records.append(run_record)
                    count += 1

    all_runs_df = pd.DataFrame(all_run_records)

    # Build aggregation dictionary dynamically based on attributes present in the dataframe
    agg_dict = {}
    # always aggregate means/std for these overall metrics
    for col in ['acc_mean', 'balanced_acc_mean', 'auc_mean']:
        if col in all_runs_df.columns:
            agg_dict[col] = ['mean', 'std', 'count']

    # for each attribute, aggregate group metrics
    for attr in attributes:
        for metric in ['acc', 'balanced_acc', 'auc', 'loss']:
            for g in ['0', '1']:
                col = f'{metric}_{attr}_{g}'
                if col in all_runs_df.columns:
                    # for loss we may want mean and std too
                    agg_dict[col] = ['mean', 'std']

    # if nothing to aggregate (defensive), return
    if not agg_dict:
        return all_runs_df, pd.DataFrame()

    summary_df = all_runs_df.groupby('experiment').agg(agg_dict)

    # flatten and keep experiment index as column
    summary_df['experiment'] = summary_df.index

    # If your experiment naming convention is the same, extract subgroup and proportion
    # e.g., 'llrt_alloc_A_20' -> subgroup 'A', proportion 20
    # This regex assumes last underscore then digits are proportion
    summary_df['subgroup'] = summary_df['experiment'].str.extract(r'alloc_(.+)_\d+$', expand=False)
    # fallback to prior naming if not matching
    if summary_df['subgroup'].isnull().any():
        # try simpler split
        summary_df['subgroup'] = summary_df['experiment'].str.split('_').str[-2]

    # proportion
    summary_df['proportion'] = summary_df['experiment'].str.split('_').str[-1].astype(float, errors='ignore')
    # count (use acc_mean count if available, else auc_mean count)
    count_col = None
    for cand in [('acc_mean', 'count'), ('balanced_acc_mean', 'count'), ('auc_mean', 'count')]:
        if cand in summary_df.columns:
            count_col = cand
            break
    if count_col:
        summary_df['count'] = summary_df[count_col]
        # drop the multiindex count entry to avoid duplication
        try:
            summary_df.drop(columns=[count_col], inplace=True)
        except Exception:
            pass

    return all_runs_df, summary_df
