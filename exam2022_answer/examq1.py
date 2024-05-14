import numpy as np
import pandas as pd
import datetime as dt

# Load and clean data
def load_windmill_data(sheetname):
    """ Loads and cleans the data from the downloaded windmill spreadsheet.
    Inputs:
    sheetname (str): Name of sheet in windmills.xlsx to be loaded. 
                     Can be 'IkkeAfmeldte-Existing turbines' or 'Afmeldte-Decommissioned'

    Returns:
    raw_df (DataFrame): Formatted dataframe.
     """

    # Load data
    raw_df = pd.read_excel('windmills.xlsx', sheet_name=sheetname, header=9)
    raw_df = raw_df.iloc[:,:-2] #drop last two columns which contain tentative data for 2022

    # Rename columns with yearly production
    col_names = {}
    for col in raw_df.columns:
        if 'Unnamed' in str(col):
            newname = 'year'+str(int(raw_df[col][6]))
            col_names[col] = newname
    raw_df.rename(columns=col_names, inplace=True)

    # Drop empty observations
    raw_df.drop(range(8), inplace=True)
    raw_df.reset_index(inplace=True, drop=True)

    # Give turbine characteristic variables more programming friendly names
    col_names = {'Turbine identifier (GSRN)': 'turbine_id',
             'Date of decommissioning': 'decom_date',
             'Date of original connection to grid': 'connect_date',
             'Capacity (kW)': 'capkW',
             'Rotor diameter (m)': 'rodiam',
             'Hub height (m)': 'heightm',
             'Type of location': 'loc_type',
             'Cadastral district': 'caddist',
             'Cadastral no.': 'caddist_no',
             'X (east) coordinate\nUTM 32 Euref89': 'Xcoor',
             'Y (north) coordinate\nUTM 32 Euref89': 'Ycoor',
             'Origin of coordinates': 'coor_orig',
             'Distribution company installation number': 'installno'}
    raw_df.rename(columns=col_names, inplace=True)

    # Format connection date
    raw_df['connect_date'] = pd.to_datetime(raw_df.connect_date)
   
    return raw_df




# 7 year moving average
def MA7(df, column):
    """ Calculates a 7 year moving average of column in df and adds it to df. 
    Inputs:
    df (DataFrame): DataFrame containing variable of interest. Index must be set to year.
    column (DataFrame series): Column in df

    Returns:
    None
    """
    # Loop over years in index
    for year in df.index:
        # Set range of years to average over
        ma_range = range(-3, 3+1)
        ma_years= [year + c for c in ma_range]

        # Check if the observation is +-3 years from endpoint
        # If so, 7 year moving average cannot be caluculated, and the year is skipped.
        in_range = True
        for y in ma_years:
            if y not in df.index:
                in_range = False
        if not in_range:
            continue            

        # Calculate MA and add to dataframe
        maname = 'MA'+column
        ma = df.loc[ma_years, column].mean()
        df.loc[year, maname ] = ma


# Simple OLS
def ols_predict(y_ser, x_ser):
    """ Simple prediction by simple linear regression of y_ser on x_ser.
    Inputs:
    y_ser (list-like): Regressand
    x_ser (list-like): Regressor

    Returns:
    yhat (array): Linear fit 
    betahat (array): Estimated OLS coefficients
    """
    # Construct matrices
    x0  = np.ones(len(y_ser))
    x1 = np.array(x_ser)
    X = np.stack((x0,x1)).T
    y = np.array(y_ser)

    # Regression and fit
    betahat = np.linalg.inv(X.T@X)@X.T@y
    yhat = X@betahat

    return yhat, betahat