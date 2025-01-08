import pandas as pd
import numpy as np
import warnings
from matplotlib.collections import PathCollection
from matplotlib.legend_handler import HandlerPathCollection, HandlerLine2D
from scipy import stats
import matplotlib.pyplot as plt
import datetime
from matplotlib.ticker import LinearLocator


def aggregate_time_series(
    data: pd.DataFrame,
    freq: str = 'D',
    ) -> pd.DataFrame:
    """Resamples each time series in the input DataFrame
    to a longer timespan and computes the average over each new timespan.

    Args:
        data (pd.DataFrame): Multiple time series with higher sampling rate.
        freq (str, optional): Frequency of the resampled time series.
        Defaults to 'D'.

    Raises:
        ValueError: _description_

    Returns:
        pd.DataFrame: _description_
    """
    # Resample each time series in the dataframe to the desired frequency
    resampled = data.resample(freq).mean()#.shift(1, freq=freq)

    # Check if the resampled data at least contains 10% of the original data
    full_length = data.groupby(pd.Grouper(freq=freq)).size().max()
    mask = data.groupby(pd.Grouper(freq=freq)).size() >= 0.2*full_length # type: ignore
    mask.index = resampled.index
    resampled['smaples_amount'] = data.groupby(pd.Grouper(freq=freq)).size()

    resampled[~mask] = np.nan
    return resampled

def format_y_ticks(value, _):
    return f'{value:.3f}'

def plot_control_charts(
    smart_tracked_modes: pd.DataFrame,
    modal_data: pd.DataFrame,
    predictions: pd.DataFrame,
    mode: str,
    start: datetime.datetime,
    end: datetime.datetime,
    confidence: float = 0.99,
    timespans: list[str] = ['D', '2D', 'W', '2W'],
    markers: list[str] = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'],
    ylim = None
    ) -> None:

    warnings.filterwarnings('ignore')

    z = stats.norm.ppf((1 + confidence) / 2)
    for timespan in timespans:

        plt.figure(figsize=(12,6))
        plt.grid()
        plt.title(mode + ' control chart')
        m = 0
        i = 0

        
        normalized = pd.DataFrame((smart_tracked_modes['frequency'] - predictions['prediction']).dropna() + smart_tracked_modes['frequency'].mean(), columns=['normalized_frequency'])

        timely_tracked_frequencies = aggregate_time_series(pd.DataFrame(smart_tracked_modes['frequency']), timespan)
        timely_normalized_frequencies = aggregate_time_series(normalized, timespan)

        std = timely_tracked_frequencies['frequency'].std()
        confidence_interval = z * std / np.sqrt(timely_tracked_frequencies['smaples_amount'])

        plt.scatter(modal_data.index, modal_data['mean_frequency'], c='tab:blue', alpha=0.5, label = 'Reference-based tracked', s=1)
        plt.scatter(smart_tracked_modes.index, smart_tracked_modes['frequency'], c='tab:green', alpha=0.5, label = 'Uncertainty tracked', s=3)
        plt.plot(timely_tracked_frequencies['frequency'], marker = markers[m], c='tab:purple')
        plt.plot(timely_normalized_frequencies['normalized_frequency'], marker = markers[m], c='tab:orange')
        plt.fill_between(timely_tracked_frequencies.index, (timely_tracked_frequencies['frequency']-confidence_interval).values, (timely_tracked_frequencies['frequency']+confidence_interval).values, label = 'averaged', color='tab:purple', alpha=0.4)
        plt.fill_between(timely_normalized_frequencies.index, (timely_normalized_frequencies['normalized_frequency']-confidence_interval).values, (timely_normalized_frequencies['normalized_frequency']+confidence_interval).values, label = 'averaged after normalization', color='tab:orange', alpha=0.4)
        plt.xlim(start, end)
        if ylim:
            plt.ylim(ylim)
        #plt.yticks([])
        plt.xlabel('Timestamp (YYYY-MM)')
        plt.ylabel('Frequency (Hz)', )

        
        def update(handle, orig):
            handle.update_from(orig)
            handle.set_alpha(1)

        legend = plt.legend(handler_map={PathCollection : HandlerPathCollection(update_func= update),
                                plt.Line2D : HandlerLine2D(update_func = update)}
                    ,loc='upper left',
                    fontsize=16)
        for lh in legend.legendHandles[:2]:
            lh.set_alpha(1)
            lh.set_sizes([30])

        plt.show()
        plt.close()

def plot_decision_charts(
    all_smart_tracked_modes: dict[str, pd.DataFrame],
    all_modal_data: dict[str, pd.DataFrame],
    all_predictions: dict[str, pd.DataFrame],
    start: datetime.datetime,
    end: datetime.datetime,
    confidence: float = 0.9,
    timespans: list[str] = ['D', '2D', 'W', '2W'],
    markers: list[str] = ['.', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'],
    ylims = None):

    warnings.filterwarnings('ignore')
    z = stats.norm.ppf((1 + confidence) / 2)

    for timespan in timespans:
        num_rows = len(all_smart_tracked_modes.keys())
        fig, axs = plt.subplots(num_rows, 1, figsize=(12, 3 * num_rows))  # Adjust the height as needed

        for i, mode in enumerate(all_smart_tracked_modes.keys()):
            normalized = pd.DataFrame(
                (all_smart_tracked_modes[mode]['frequency'] - all_predictions[mode]['prediction']).dropna() \
                + all_smart_tracked_modes[mode]['frequency'].mean(),
                columns=['normalized_frequency'])

            aggregated_frequencies = aggregate_time_series(pd.DataFrame(all_modal_data[mode]['mean_frequency']), timespan)
            timely_normalized_frequencies = aggregate_time_series(normalized, timespan)

            std = all_modal_data[mode]['mean_frequency'].std()
            confidence_interval = z * std / np.sqrt(aggregated_frequencies['smaples_amount'])
            std_smart = normalized['normalized_frequency'].std()
            confidence_interval_smart = z * std_smart / np.sqrt(timely_normalized_frequencies['smaples_amount'])

            axs[i].scatter(all_modal_data[mode].index, all_modal_data[mode]['mean_frequency'], c='tab:blue', alpha=1.0, label = 'reference-based tracked', s=1)
            axs[i].scatter(all_smart_tracked_modes[mode].index, all_smart_tracked_modes[mode]['frequency'], c='tab:orange', alpha=1.0, label = 'smart tracked', s=1)
            axs[i].plot(aggregated_frequencies['mean_frequency'], marker = markers[2], c='tab:purple')
            axs[i].plot(timely_normalized_frequencies['normalized_frequency'], marker = markers[2], c='tab:green')
            axs[i].fill_between(
                aggregated_frequencies.index,
                (aggregated_frequencies['mean_frequency']-confidence_interval).values,
                (aggregated_frequencies['mean_frequency']+confidence_interval).values,
                label = 'weekly average',
                color='tab:purple',
                alpha=0.4)
            axs[i].fill_between(
                timely_normalized_frequencies.index,
                (timely_normalized_frequencies['normalized_frequency']-confidence_interval_smart).values,
                (timely_normalized_frequencies['normalized_frequency']+confidence_interval_smart).values,
                label = 'weekly average, normalized',
                color='tab:green',
                alpha=0.4)
            
            axs[i].set_xlim(start, end)
            if ylims:
                axs[i].set_ylim(ylims[mode])
            axs[i].set_ylabel(mode + ' (Hz)')

            axs[i].grid()
            numSteps = 5
            axs[i].yaxis.set_major_locator(LinearLocator(numSteps))
            yticks = axs[i].yaxis.get_major_ticks()
            axs[i].yaxis.set_major_formatter(plt.FuncFormatter(format_y_ticks))

            axs[i].hlines(all_smart_tracked_modes[mode]['frequency'].mean()*1.01, start, end, color='k', linestyle='--', label = 'anomaly threshold')
            axs[i].hlines(all_smart_tracked_modes[mode]['frequency'].mean()*0.99, start, end, color='k', linestyle='--')
            
            if yticks:
                yticks[-1].label1.set_visible(False)
                yticks[0].label1.set_visible(False)
            if i == 1:
                legend = axs[0].legend(loc='upper left', fontsize=12, bbox_to_anchor=(0.0, 1.4))
                for lh in legend.legendHandles[:2]:
                    lh.set_alpha(1)
                    lh.set_sizes([30])
            
            if i != num_rows-1:
                axs[i].set_xticklabels([])  # This will remove the x-axis values
            else:
                axs[i].set_xlabel('Timestamp (YYYY-MM)')

                
                xticks = axs[i].xaxis.get_major_ticks()
                for tick in xticks[::2]:
                    tick.label1.set_visible(False)

        fig.subplots_adjust(hspace=0.02)
        fig.show()
        plt.show()
        plt.close()







