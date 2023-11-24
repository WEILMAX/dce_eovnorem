import pandas as pd
import numpy as np

def sin_cos_angle_inputs(
    inputs: pd.DataFrame,
    angle_type: str = 'degrees',
    angles: list[str] = ['winddirection', 'yaw']
    ) -> pd.DataFrame:
    """This function takes in a pandas DataFrame
    containing numerical inputs and returns a new DataFrame 
    that includes the sine and cosine transform 
    of any input column that contains angles.
    The input DataFrame should not contain any 
    non-numerical columns.
    
    Parameters:
    inputs (pandas DataFrame): The input DataFrame containing numerical inputs.
    angle_type (str): The unit in which angles are represented in the input DataFrame.
        Defaults to 'degrees'.
        Valid options are 'degrees' or 'radians'.
    
    Returns:
    pandas DataFrame: A new DataFrame that includes the sine and cosine transform 
        of any input column that contains angles.
        The original DataFrame is not modified.
    """
    angle_cols = [col_name for col_name in inputs.columns if any(angle in col_name for angle in angles)]

    outputs = inputs.copy()
    for col in angle_cols:
        if angle_type == 'degrees':
            outputs[col] = np.deg2rad(inputs[col])
        outputs['sin_' + col] = np.sin(outputs[col])
        outputs['cos_' + col] = np.cos(outputs[col])
    outputs.drop(columns=angle_cols, inplace=True)
    return outputs