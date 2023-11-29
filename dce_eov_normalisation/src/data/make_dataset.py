import pandas as pd
from src.data.utils import *
from src.data.preprocessing import *


def get_turbine_data(data_folder: str, location: str, turbine: str) -> pd.DataFrame:
    """Get the turbine data from the location folgder in the raw data folder.

    Args:
        data_folder (str): Folder containing all the data.
        location (str): Location of the turbine.
        turbine (str): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        pd.DataFrame: _description_
    """    
    file_path = os.path.join(data_folder, 'raw', location, location + turbine + '.parquet')
    print(file_path)
    try:
        turbine_data = pd.read_parquet(file_path)
        return turbine_data
    except Exception as e:
        # Raise a FileNotFoundError if an error occurs
        raise FileNotFoundError(
            f'Turbine data not found at {file_path}. ',
            'Import the data first from owi_data_2_pandas API.\nOriginal Error: {e}'
        ) from e

def get_mvbc_data(data_folder: str, location: str) -> pd.DataFrame:
    """Get the mvbc data from the location folder in the external data folder.

    Args:
        data_folder (_type_): _description_
        location (_type_): _description_

    Raises:
        FileNotFoundError: _description_

    Returns:
        _type_: _description_
    """    
    file_path = os.path.join(data_folder, 'external', location, 'mvbc_data.parquet')
    try:
        mvbc_data = pd.read_parquet(file_path)
        return mvbc_data
    except Exception as e:
        # Raise a FileNotFoundError if an error occurs
        raise FileNotFoundError(
            f'Turbine data not found at {file_path}. ',
            'Import the data first from mvbc API.\nOriginal Error: {e}'
        ) from e
    

def create_input_target_dataset(
    data_folder: str,
    location: str,
    turbine: str,
    mode: str
    ) -> tuple[pd.DataFrame, pd.Series]:
    """_summary_

    Args:
        data_folder (str): _description_
        location (str): _description_
        turbine (str): _description_
        mode (str): _description_

    Returns:
        tuple[pd.DataFrame, pd.Series]: _description_
    """
    turbine_data_ = pd.read_parquet(os.path.join(data_folder, 'raw', location, location + turbine + '.parquet'))
    scada_means = select_mean_SCADA(turbine_data_)
    scada_means_selected = scada_means.loc[:, scada_means.isna().sum() < 0.05 * scada_means.shape[0]]
    scada_means_preprocessed = sin_cos_angle_inputs(scada_means_selected)
    if len(set(scada_means.columns) - set(scada_means_selected.columns)) > 0:
        print('dropped scada columns: ', set(scada_means.columns) - set(scada_means_selected.columns))

    mvbc_data = get_mvbc_data(data_folder, location)
    mvbc_data_selected = mvbc_data.loc[:, mvbc_data.isna().sum() < 0.05 * scada_means.shape[0]]
    if len(set(mvbc_data.columns) - set(mvbc_data_selected.columns)) > 0:
        print('dropped mvbc columns: ', set(mvbc_data.columns) - set(mvbc_data_selected.columns))


    synced_data = synchronize_data({
        'SCADA_means': scada_means_preprocessed,
        'mvbc_data_selected': mvbc_data
        }
    )

    mode_ = get_reference_based_mode(data_folder, mode, location, turbine)
    model_input, model_target = create_input_output(synced_data, mode_)
    return model_input, model_target
