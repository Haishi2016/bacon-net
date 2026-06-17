from fire import Fire
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_robustness(metrics_path):
    df = pd.read_csv(metrics_path).dropna()
    fig, ax = plt.subplots()
    rescaled_labels_acc = (df['label_acc_mean'] - df['label_acc_mean'].min()) / (
            df['label_acc_mean'].max() - df['label_acc_mean'].min())
    rescaled_labels_acc_std = df['label_acc_std'] / (
            df['label_acc_mean'].max() - df['label_acc_mean'].min())

    ax.plot(rescaled_labels_acc, label='Labels accuracy')
    plt.fill_between(range(len(rescaled_labels_acc)), rescaled_labels_acc - rescaled_labels_acc_std,
                     rescaled_labels_acc + rescaled_labels_acc_std, alpha=0.5)

    rescaled_concepts_acc = (df['concepts_acc_mean'] - df['concepts_acc_mean'].min()) / (
            df['concepts_acc_mean'].max() - df['concepts_acc_mean'].min())
    rescaled_concepts_acc_std = df['concepts_acc_std'] / (
            df['concepts_acc_mean'].max() - df['concepts_acc_mean'].min())

    ax.plot(rescaled_concepts_acc, label='Concepts accuracy')
    plt.fill_between(range(len(rescaled_concepts_acc)),
                     rescaled_concepts_acc - rescaled_concepts_acc_std,
                     rescaled_concepts_acc + rescaled_concepts_acc_std, alpha=0.5)
    ax.set_xlabel('Noise level')
    ax.set_ylabel('Min-max rescaled value')


if __name__ == '__main__':
    Fire(plot_robustness)
