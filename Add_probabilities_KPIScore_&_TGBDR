import pandas as pd
import numpy as np

def add_combined_probabilities(df, kpiscore_column='KPIScore', tgbdr_column='TGBDR', age_column='Age'):
    """
    Add pregnancy probabilities based on both KPIScore and TGBDR+Age combination.
    All probabilities are in decimal format (0-1 range).
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    kpiscore_column (str): Name of the column containing KPIScore values
    tgbdr_column (str): Name of the column containing TGBDR values
    age_column (str): Name of the column containing Age values
    
    Returns:
    pandas.DataFrame: DataFrame with added probability columns
    """
    # KPIScore probability mapping
    kpi_prob_mapping = {
        25: 0.70, 24: 0.65, 23: 0.60, 22: 0.55, 21: 0.50,
        20: 0.45, 19: 0.40, 18: 0.35, 17: 0.30, 16: 0.26,
        15: 0.22, 14: 0.18, 13: 0.15, 12: 0.13, 11: 0.10,
        10: 0.09, 9: 0.07, 8: 0.06, 7: 0.05, 6: 0.04, 5: 0.03
    }

    # KPIScore confidence interval mapping
    kpi_ci_mapping = {
        25: (0.59, 0.79), 24: (0.56, 0.74), 23: (0.51, 0.69), 22: (0.47, 0.63),
        21: (0.42, 0.57), 20: (0.38, 0.51), 19: (0.33, 0.46), 18: (0.28, 0.41),
        17: (0.24, 0.36), 16: (0.20, 0.32), 15: (0.16, 0.30), 14: (0.13, 0.26),
        13: (0.10, 0.22), 12: (0.08, 0.20), 11: (0.06, 0.18), 10: (0.05, 0.15),
        9: (0.04, 0.14), 8: (0.03, 0.12), 7: (0.02, 0.10), 6: (0.016, 0.09),
        5: (0.012, 0.08)
    }

    # TGBDR-Age probability mapping
    tgbdr_age_mapping = {
        ('<40%', '<38'): 0.308,
        ('<40%', '38-39'): 0.243,
        ('<40%', '>40'): 0.140,
        ('40-50%', '<38'): 0.611,
        ('40-50%', '38-39'): 0.540,
        ('40-50%', '>40'): 0.333,
        ('>50%', '<38'): 0.697,
        ('>50%', '38-39'): 0.644,
        ('>50%', '>40'): 0.426
    }

    def get_tgbdr_category(tgbdr):
        if tgbdr < 40:
            return '<40%'
        elif tgbdr <= 50:
            return '40-50%'
        else:
            return '>50%'

    def get_age_category(age):
        if age < 38:
            return '<38'
        elif age <= 39:
            return '38-39'
        else:
            return '>40'

    # Create a copy of the dataframe
    df_new = df.copy()

    # Add KPIScore-based probabilities
    df_new['Prob_KPI'] = df_new[kpiscore_column].map(kpi_prob_mapping).astype(float)
    df_new['CI_Lower'] = df_new[kpiscore_column].map(lambda x: kpi_ci_mapping.get(x, (np.nan, np.nan))[0]).astype(float)
    df_new['CI_Upper'] = df_new[kpiscore_column].map(lambda x: kpi_ci_mapping.get(x, (np.nan, np.nan))[1]).astype(float)

    # Add TGBDR and Age categories
    df_new['TGBDR_Category'] = df_new[tgbdr_column].apply(get_tgbdr_category)
    df_new['Age_Category'] = df_new[age_column].apply(get_age_category)

    # Add TGBDR-Age based probability
    df_new['Prob_TGBDR_Age'] = df_new.apply(
        lambda row: tgbdr_age_mapping.get(
            (row['TGBDR_Category'], row['Age_Category']), np.nan
        ),
        axis=1
    )

    # Round all probability values to 3 decimal places
    probability_columns = ['Prob_KPI', 'CI_Lower', 'CI_Upper', 'Prob_TGBDR_Age']
    for col in probability_columns:
        df_new[col] = df_new[col].round(3)

    return df_new

df_with_probabilities = add_combined_probabilities(
    df,
    kpiscore_column='KPIScore',  
    tgbdr_column='TGBDR',        
    age_column='Age'             
)
df_with_probabilities.to_excel('Combined_Probabilities.xlsx', float_format='%.3f')
