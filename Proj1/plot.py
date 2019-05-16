import pickle

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_from_tensors(accuracies, losses, model_names, save_df=True):
    """
    Create and save plots 4 from tensors with confidentiality intervals,
    one for each phase (train, test) and each data (loss, accuracy)
    :param accuracies: dict, contain 2 tensors representing the accuracy per epoch, on for each phase (train, test)
    :param losses: dict, contain 2 tensors representing the loss per epoch, on for each phase (train, test)
    :param model_names: list, name of the models
    :param save_df: bool, whether to save the dataframe constructed with the tensors
    """
    def to_pandas(loss, acc, model_name):
        df_all = pd.DataFrame(data=None, columns=['model', 'phase', 'epoch', 'acc', 'loss'])
        for phase in ['train', 'test']:
            for loss_round, acc_round in zip(loss[phase], acc[phase]):
                data = zip(range(len(loss_round)), loss_round.numpy(), acc_round.numpy())
                df = pd.DataFrame(data=data, columns=['epoch', 'loss', 'acc'])
                df['phase'] = phase
                df['model'] = model_name
                df_all = pd.concat((df_all, df), sort=True)
        return df_all

    stats_df = pd.DataFrame(data=None, columns=['model', 'phase', 'epoch', 'acc', 'loss'])
    for loss, acc, name in zip(losses, accuracies, model_names):
        df = to_pandas(loss, acc, name)
        stats_df = pd.concat((stats_df, df), sort=True)

    if save_df:
        pickle.dump(stats_df, open('stats_df.p', mode='wb'))

    plot_stats(stats_df)


def plot_stats(stats_df=None, ext='png', figsize=(16, 9)):
    """
    Create and save plots 4 from tensors with confidentiality intervals,
    one for each phase (train, test) and each data (loss, accuracy)
    :param stats_df: pandas.DataFrame, the dataframe containing the all the data for the plots
    :param ext: str, the extension for the saved plots ('svg' or 'png' for example)
    :param figsize: tuple(int, int), the size of the plots
    """
    if stats_df is None:
        stats_df = pickle.load(open('stats_df.p', mode='rb'))

    def save_plot(stats, phase):
        filename = f'{stats}_{phase}.{ext}'
        title = f'{stats.capitalize()} ({phase})'
        plt.figure(figsize=figsize)
        plt.title(title)
        data = stats_df[stats_df['phase'] == phase]
        sns.lineplot(x="epoch", y=stats, hue='model', data=data)
        plt.savefig(filename)

    for stats in ['acc', 'loss']:
        for phase in ['train', 'test']:
            save_plot(stats, phase)
