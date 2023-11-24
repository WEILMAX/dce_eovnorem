import pandas as pd

def harmonics(rpm_data:pd.Series, p_orders:list):
    harmonic_data = pd.DataFrame(index = rpm_data.index)
    for p_order in p_orders:
        harmonic_data[f'harmonic_{p_order}p'] = (p_order/60) * rpm_data
    return harmonic_data