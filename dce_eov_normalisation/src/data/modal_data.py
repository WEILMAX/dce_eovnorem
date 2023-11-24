import pandas as pd

def get_mpe_data(
    turbine_data:pd.DataFrame,
    mode_direction:str,
    location:str,
    turbine:str,
    ) -> pd.DataFrame:
    """Get the mpe data for a given mode and direction.

    Args:
        data (pd.DataFrame): dataset from the turbine containg SCADA anjd SHM data.
        mode_direction (str): Mode direction, SS (Side-Side) or Fore-Aft (FA).
        location (str): Location of the turbine, windfarm acronym (e.g. nrt, nw2,...).
        turbine (str): Code of the windturbine inside the windfarm (e.g. c02, c03,...).

    Raises:
        ValueError: mode_direction must be FA or SS.

    Returns:
        pd.DataFrame: mpe data for the given mode and direction.
    """    
    name_location = ('_'.join([location, turbine])).upper()
    mpe_data = turbine_data.filter(regex='mpe')
    if mode_direction not in ['FA', 'SS']:
        raise ValueError('mode_direction must be FA or SS')
    mpe_direction = \
        pd.DataFrame.from_records(
            mpe_data['_'.join(['mpe', name_location, mode_direction])].explode().dropna().tolist()
        )\
            .set_index(
                mpe_data['_'.join(['mpe', name_location, mode_direction])].explode().dropna().index
            )
    return mpe_direction