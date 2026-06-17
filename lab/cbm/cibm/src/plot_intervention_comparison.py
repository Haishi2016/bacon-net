from ast import literal_eval

import numpy as np
from matplotlib import pyplot as plt


def plot_one_result(ax, data, label, plot_std=True):
    acc_tti = [(k, np.mean(acc), np.std(acc)) for k, acc in data]
    acc_tti.sort()
    acc_tti_means = np.array([m for _, m, _ in acc_tti])
    acc_tti_stds = np.array([std for _, _, std in acc_tti])
    ax.plot(acc_tti_means, label=label)
    if plot_std:
        plt.fill_between(range(len(acc_tti_means)), acc_tti_means - acc_tti_stds,
                         acc_tti_means + acc_tti_stds, alpha=0.5)
    auc = acc_tti_means.mean()
    print(f"Intervention AUC for {label} = {auc}")
    normalized_auc = 0
    prev_value = acc_tti[0][1]
    for i, m, s in acc_tti[1:]:
        normalized_auc += (m - prev_value)
        prev_value = m
    normalized_auc /= len(acc_tti)
    print(f"Normalized Interventions AUC = {normalized_auc}")
    return ax


def plot(path_1, path_2, label_1, label_2, result_path):
    fig, ax = plt.subplots()
    for path, label in [(path_1, label_1), (path_2, label_2)]:
        with open(path, 'r') as f:
            lines = [tuple(line.strip().split()) for line in f.readlines()]
        data = [(int(line[0]), literal_eval(' '.join(line[1:]))) for line in lines]

        ax = plot_one_result(ax, data, label)

    ax.set_xlabel('Number of intervened groups (random strategy)')
    ax.set_ylabel('Target accuracy')
    ax.legend()
    plt.savefig(result_path)


if __name__ == '__main__':
    log_id = input()
    
    plot(f'logs/{log_id}/interventions.txt',
         f'logs/{log_id}/interventions_uncertainty.txt',
         'random',
         'uncertaintly',
         f'logs/{log_id}/all_interventions_cub_100_epochs.pdf')
