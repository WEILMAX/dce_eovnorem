import os
import pandas as pd
import numpy as np

def get_input_data(data_folder: str, location: str, turbine: str):
    """Get all the data for a given turbine at a given location

    Args:
        data_folder (str): data folder
        location (str): location
        turbine (str): turbine

    Returns:
        _type_: _description_
    """    
    data = {}
    data['turbine_data'] = pd.read_parquet(os.path.join(data_folder, 'raw', location, location + turbine + '.parquet'))
    data['mvbc_data'] = pd.read_parquet(os.path.join(data_folder, 'external', location, 'mvbc_data.parquet'))
    
    #data['SS1_initial'] = pd.read_parquet(os.path.join(data_folder, 'processed', location, 'tracked_modes', 'SS1_'+turbine_code+'.parquet'))
    #data['SS2_initial'] = pd.read_parquet(os.path.join(data_folder, 'processed', location, 'tracked_modes', 'SS2_'+turbine_code+'.parquet'))
    return data

def get_reference_based_mode(data_folder:str, mode:str, location: str, turbine: str):
    turbine_code = (location + turbine).upper()
    mode_data = pd.read_parquet(os.path.join(data_folder, 'interim', location, 'tracked_modes', 'reference_based', mode+'_'+turbine_code+'.parquet'))
    return mode_data

def select_duplicated_modes(mode_data: pd.DataFrame):
    """Drop modes with duplicated index in the mode data and only keep the mode with the biggest value in the size column.

    Args:
        mode_data (pd.DataFrame): _description_
    """    
    selected_mode_data = mode_data.copy()
    selected_mode_data = selected_mode_data.sort_values(by=['size'], ascending=False)
    selected_mode_data = selected_mode_data[~selected_mode_data.index.duplicated(keep='first')]
    selected_mode_data = selected_mode_data.sort_index()
    return selected_mode_data
    

def synchronize_data(data: dict[str, pd.DataFrame]):
    """Synchronize the dataframes in the data dictionnary.

    Args:
        data (dict[str, pd.DataFrame]): dictionnary of dataframes
    """
    synced_data = pd.DataFrame(index=data[list(data.keys())[0]].index)
    for key in data.keys():
        data_key_ = data[key].copy()
        data_key_.index = pd.to_datetime(data_key_.index)
        if len(data_key_.index) < len(synced_data.index):
            # interpolate the data if less granularity than the first dataset (turbine data)
            data_key_ = data_key_.reindex(synced_data.index, method='nearest')
        synced_data = pd.concat([synced_data, data_key_], axis=1)
    return synced_data

def select_mean_SCADA(data: pd.DataFrame):
    """Select the column in a dataframe that contain the mean

    Args:
        data (pd.DataFrame): dataframe
        column (str): column name

    Returns:
        pd.DataFrame: dataframe with the means
    """    
    data_means_ = data.filter(regex='mean')
    SCADA_means = data_means_.drop(columns = data_means_.filter(regex = 'acc|ACC').columns)
    return SCADA_means

def create_input_output(synced_data: pd.DataFrame, mode:pd.DataFrame):
    # Only keep the mode with biggest size when duplicates
    mode_selected_ = select_duplicated_modes(mode)
    mode_frequency_ = mode_selected_['frequency']
    inputs_ = synced_data.copy()
    y = mode_frequency_.dropna()
    X = inputs_.loc[y.index].dropna()
    y = y.loc[X.index]
    return X, y