import matplotlib.pyplot as plt
from mlproject.evaluate import util

def plot_single_metric(ax, train_df, val_df, metric, metric_title):
    train_metric_name = 'train/' + str(metric)
    val_metric_name = 'val/' + str(metric)
    ax.plot(train_df['step'], train_df[train_metric_name], '-o', label=train_metric_name)
    ax.plot(val_df['step'], val_df[val_metric_name], '-o', label=val_metric_name)
    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel(metric_title, fontsize=14)
    ax.set_title(metric_title, fontsize=16, fontweight='bold')
    ax.legend(fontsize=14); ax.grid()
    plt.setp(ax.get_xticklabels(), fontsize=14)
    plt.setp(ax.get_yticklabels(), fontsize=14)

def plot_two_metrics(train_df, val_df, metrics, metric_titles):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plot_single_metric(fig.axes[0], train_df, val_df, metrics[0], metric_titles[0])
    plot_single_metric(fig.axes[1], train_df, val_df, metrics[1], metric_titles[1])
    return fig, ax


def main():
    metrics_csv, save_file = util._setup_parser_get_args()
    train_df, val_df = util.prepare_dataframes_from_csv(metrics_csv=metrics_csv)
    fig, ax = plot_two_metrics(train_df, val_df, metrics=('loss', 'accu'), metric_titles=('Loss', 'Accuracy'))
    plt.show()
    if save_file is not None:
        plt.savefig(save_file)


if __name__ == '__main__':
    main()