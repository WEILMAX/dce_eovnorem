import pandas as pd
import numpy as np


def smart_tracking(
    calculated_mode: pd.Series,
    predictions: pd.Series,
    uncertainties: pd.Series,
    uncertainty_threshold: float,
    prediction_lim: float
    ) -> pd.DataFrame:
    # only keep uncertainties for which we have a calculated_mode
    common_indices = uncertainties.index.intersection(calculated_mode.index)
    uncertainties_common = uncertainties.loc[common_indices]

    # Filter based on uncertainty threshold
    filtered_modes = calculated_mode.loc[uncertainties_common[uncertainties_common < uncertainty_threshold].index]
    fitered_modes = pd.DataFrame(filtered_modes.loc[filtered_modes.index.intersection(calculated_mode.index)])
    differences_to_prediction = pd.DataFrame(np.abs(filtered_modes - predictions).dropna(), columns=['difference'])

    result_ = pd.concat([fitered_modes, differences_to_prediction], axis=1, join='outer')
    result = result_.copy()
    result.sort_values(by='difference', ascending=True, inplace=True)
    # ensure unique index and keep first occurence
    result = result[~result.index.duplicated(keep='first')]
    result.sort_index(inplace=True)
    result.rename(columns={'mean_frequency': 'frequency'}, inplace=True)
    result = result[result['difference'] < prediction_lim]
    return result