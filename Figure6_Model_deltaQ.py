#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:03:30 2024

@author: butl842
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 

## Stream id
stream_id = "HOPB" # Stream / Precip: a) HOPB / HARV, b)  KING / KONZ, c) LECO / GRSM, d) MART / WREF, e) MCDI / KONZ, f) SYCA / SYCA
precip_id = "HARV"

stream_list = "HOPB", "KING", "LECO", "MART", "MCDI", "SYCA"
# Precip , Stream
# pairs = [['a) ARIK','ARIK'],['b) BLAN','LEWI'],['c) BLUE','BLUE'], ['d) BONA','CARI'],['e) CLBJ','PRIN'],['f) CUPE','CUPE'],
#             ['g) DELA','BLWA'],['h) GRSM','LECO'],['i) GUIL','GUIL'],['j) HARV','HOPB'],['k) JERC','FLNT'],['l) KONZ','KING'],['m) KONZ','MCDI'],       
#             ['n) LENO','TOMB'],['o) NIWO','COMO'],['p) ORNL','WALK'],['q) REDB','REDB'],['r) SCBI','POSE'],['s) SYCA','SYCA'],['t) TALL','MAYF'],
#             ['u) TOOL','OKSR'],['v) TOOL','TOOK'],['w) WOOD','PRLA'],['x) WOOD','PRPO'],['y) WREF','MART'],['z) YELL','BLDE']] 

path_obs  = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/NEONStreamflow/NEON_isotopes/'
path_2016 = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/TracerStreamflow/'
path_2018 = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/TracerStreamflow/Daily_2018/'
path_deltaQ = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/Model_DeltaQ/'
path_deltaQ_All = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/Model_DeltaQ/Daily/'
path_MTT    = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/MTT_TimeSeries/'
path_threshold = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/MassSum/Threshold/'

precip_iso    = pd.read_csv(path_obs + 'Weighted_p_daily_d18O_2016.csv')
tracer_weight =  pd.read_csv(path_2016 + f"{stream_id}_weighted_tracer_daily_2016.csv") ## Model TTD
# tracer_weight =  pd.read_csv(path_2018 + f"{stream_id}_weighted_tracer_daily_2018.csv") 
stream_iso    = pd.read_csv(path_obs + 'Stream_d18O_noOutliers_2016.csv')
stream_iso.replace(-9999, np.nan, inplace=True)

## Obs precip iso
precip_iso_obs = precip_iso[f'{precip_id}'] #Select select precip site to use

## Obs stream iso
stream_iso['Date'] = pd.to_datetime(stream_iso['Date'])
stream_iso.set_index('Date', inplace=True) # Set 'Date' as the index

# Group by year-month and find the mean of each month
stream_iso_monthly_mean = stream_iso.resample('D').mean()
stream_iso_monthly_mean.reset_index(inplace=True) # Reset index to make 'Date' a column again
obs_stream_Q = stream_iso_monthly_mean.reindex(columns=stream_list)

#%%
## Initialize a dictionary to store the first non-zero index for each column
first_nonzero_indices = {}

# Initialize a variable to keep track of the previous non-zero index
previous_nonzero_index = None

# Iterate over each column (excluding the 'Days' column if it exists)
for column in tracer_weight.columns:
    if column != 'Days':  # Exclude the 'Days' column
        # Check if there are any non-zero values in the column
        non_zero_values = tracer_weight[column][tracer_weight[column] != 0].dropna()
        
        # Find the first non-zero index if there are any non-zero values
        if len(non_zero_values) > 0:
            first_nonzero_index = non_zero_values.index[0]
            # Update the previous non-zero index
            previous_nonzero_index = first_nonzero_index
        else:
            # Use the previous non-zero index if the column is all zeros
            first_nonzero_index = previous_nonzero_index
        
        # Store the index in the dictionary
        first_nonzero_indices[column] = first_nonzero_index

# Convert dictionary values to a list
Days_lst = list(first_nonzero_indices.values())

#%%
## Create model deltaQ by multiplying streamflow weights by deltaP obs
## I want precip_iso_obs(1)*tracer_weight(1) , [precip_iso_obs(1)*tracer_weight(1) + precip_iso_obs(2)*tracer_weight(2)], 
# [precip_iso_obs(1)*tracer_weight(1) + precip_iso_obs(2)*tracer_weight(2), [precip_iso_obs(3)*tracer_weight(3)]

cumulative_products = []
Mean_input_time = []
Fyw_time = []

## Loop through 
for i in np.arange(2557): # 60 months, 1826 days, 2557 days 
    model_trace = tracer_weight.values[i, :(i+1)] # Get each weighted model tracer value
    deltaP = precip_iso_obs[:(i+1)].values # Nan not 0.
    
    ## Each tracer should have a flux weighted deltaP
    start_days = Days_lst[:(i+1)]
    
    ## Calculate the cumulative product to find the deltaQ based on deltaP
    cumulative_product = np.nansum(deltaP * model_trace)
    cumulative_products.append(cumulative_product)
    
    ## Calculate the cumulative product to find the MTT based on weigted tracer and time
    cumulative_time = np.nansum(start_days * model_trace)
    Mean_input_time.append(cumulative_time)

deltaQ = np.array(cumulative_products) # move list in df
df_deltaQ = pd.DataFrame(deltaQ, columns=[stream_id])

Time = np.array(Mean_input_time) # move list in df
Time_df = pd.DataFrame(Time, columns=[stream_id])

# df_deltaQ.to_csv(path_deltaQ_All + f"{stream_id}_deltaQ.csv", index=False)
# Time_df.to_csv(path_MTT + f"{stream_id}_MTT.csv", index=False)


#%%
## Merge all model Q and obs Q dfs together as well as MTT time series
deltaQ_files = glob.glob(path_deltaQ_All + "/*.csv") # Load all files
MTT_files    = glob.glob(path_MTT + "/*.csv") # Load all files

# Initialize empty lists to store DataFrames
deltaQ_dfs = []
MTT_dfs = []

# Read and append CDF_files files
for file_path in deltaQ_files:
    df_deltaQ_merge = pd.read_csv(file_path)
    deltaQ_dfs.append(df_deltaQ_merge)
        
# Merge MTT files into one DataFrame
deltaQ_merged = pd.concat(deltaQ_dfs, axis=1)

# Read and append CDF_files files
for file_path in MTT_files:
    df_MTT_merge = pd.read_csv(file_path)
    MTT_dfs.append(df_MTT_merge)
        
# Merge MTT files into one DataFrame
MTT_merged = pd.concat(MTT_dfs, axis=1)
MTT_merged = MTT_merged.sort_index(axis=1)

# Sort columns in alphabetical order
deltaQ_merged = deltaQ_merged.sort_index(axis=1)
deltaQ_merged.replace(0, np.nan, inplace=True) # Replace 0 with np.nan

# deltaQ_merged.to_csv(path_deltaQ + 'Model_deltaQ.csv', index=False)

#%%

# Sort columns of MTT_merged in alphabetical order
MTT_merged = MTT_merged.sort_index(axis=1)

## Find the ranges in model MTT
# Create a dictionary to store the ranges for each column
time_change_ranges = {}

## Create a date range from January 2016 to December 2022
date_range = pd.date_range(start='2016-01-01', end='2022-12-31')

## Compare times!
# model_time = MTT_merged.values # /365 + 2018
obs_time   = date_range #stream_iso_monthly_mean['Date']

# Create a reference datetime object for the first day in 2018
first_day_2016 = pd.Timestamp('2016-01-01')
# Calculate the timedelta from the first day in 2018 for each datetime object
timedelta_from_2016 = obs_time - first_day_2016
# Convert timedelta values to days
timedelta_days = timedelta_from_2016 / np.timedelta64(1, 'D')

time_change_lst = []
time_change_fyw_lst = []
# Iterate through the columns of MTT_merged
for i, column in enumerate(MTT_merged.columns):
    
    # Calculate the time change for the current column
    time_change = (timedelta_days.values - MTT_merged[column].values) # /365 # Puts into years
    time_change_lst.append(time_change)
    
column_names = MTT_merged.columns
df_time_change = pd.DataFrame(time_change_lst).transpose()
df_time_change.columns = column_names

save_MTT    = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/'
# df_time_change.to_csv(save_MTT + 'MTT_TimeSeries.csv', index=False)

#%%
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import hydroeval as he
from matplotlib.lines import Line2D

## Read in threshold data
tracer_threshold_above = pd.read_csv(path_threshold + 'Tracer_index_threshold_above.csv')
tracer_threshold_below = pd.read_csv(path_threshold + 'Tracer_index_threshold_below.csv')
TotalQ_threshold_above = pd.read_csv(path_threshold + 'TotalQ_index_threshold_above.csv')
TotalQ_threshold_below = pd.read_csv(path_threshold + 'TotalQ_index_threshold_below.csv')

# Create a date formatter to format the x-tick labels as years
date_formatter = mdates.DateFormatter('%Y')

# Set the major locator for x-ticks to be each year (interval=1 year)
year_locator = mdates.YearLocator()
date_range = pd.date_range(start='2016-01-01', end='2022-12-31')

# Get common column names between obs_stream_Q and deltaQ_merged
common_columns = obs_stream_Q.columns.intersection(deltaQ_merged.columns)

site_names = ['a) HOPB','b) KING','c) LECO','d) MART','e) MCDI','f) SYCA']
# Create subplots
fig, axs = plt.subplots(6, 1, sharex=True, sharey=True, dpi=500, figsize=(8, 12))
axs = axs.ravel()
kge_legend_labels = []
delta_above_lst = []
delta_below_lst = []
deltaQ_threshold_lst = []
obs_data_list = []
mod_data_list = []
# Define the start index for excluding the first year of data
start_index = 0  # Assuming monthly data
deltaQ_merged_threshold = deltaQ_merged.copy() # Mkae copy to acess both
for i, column in enumerate(common_columns):
    
    # Plot the observation data
    obs_data = obs_stream_Q[column][start_index:]
    # obs_scatter = axs[i].scatter(obs_data.index, obs_data, label='Observed $\delta^{18}$O (‰)', alpha=0.7, marker='o', color='tab:blue', zorder=0)
    obs_scatter = axs[i].scatter(date_range, obs_data, label='Observed $\delta^{18}$O (‰)', alpha=0.7, marker='o', color='tab:blue', zorder=0)
    
    # For the "above" case
    values_above = tracer_threshold_above[column].values
    index_above = pd.Index(values_above).dropna()
    # deltaQ_above = deltaQ_merged[column].loc[index_above]
    
    # For the "above" case using TotalQ_threshold_above
    values_above_TotalQ = TotalQ_threshold_above[column].values
    index_above_TotalQ = pd.Index(values_above_TotalQ).dropna()
    
    # Combine both indexes
    common_indexes = index_above.intersection(index_above_TotalQ)
    
    ## Make below threshold values Nan
    # Get the DataFrame for the current column
    column_data = deltaQ_merged_threshold[column]

    # Create a mask where the index is not in the above index
    not_in_above_index_mask = ~column_data.index.isin(common_indexes)
    
    # Set the values in deltaQ_merged to NaN where the mask is True
    column_data.loc[not_in_above_index_mask] = pd.NA
    
    # Update the column in deltaQ_merged with the modified data
    deltaQ_merged_threshold[column] = column_data
    deltaQ_threshold_lst.append(deltaQ_merged_threshold)
    
    ## Define the window size for the rolling mean
    # window_size = 7 # Bi-weekly running mean to compare to bi-weekly observations
    # Calculate the rolling mean for each column in deltaQ_merged
    # running_mean_deltaQ_above = deltaQ_merged_threshold.rolling(window=window_size, min_periods=1).mean() # running_mean_deltaQ_above
    # running_mean_deltaQ_below = deltaQ_merged.rolling(window=window_size, min_periods=1).mean() # running_mean_deltaQ_below
    # above_threshold = axs[i].plot(running_mean_deltaQ_above[column], linestyle='-', label=r'$\geq$ 75%', color='darkred', zorder=2)
    # below_threshold = axs[i].plot(running_mean_deltaQ_below[column], linestyle='-', label=r'< 75% Traced Stream Water Fraction', color='lightcoral', zorder=1)
    
    above_threshold = axs[i].scatter(date_range, deltaQ_merged_threshold[column], linestyle='-', label=r'$\geq$ 75%', color='darkred', 
                                     zorder=2, s=2)
    below_threshold = axs[i].scatter(date_range, deltaQ_merged[column], linestyle='-', label=r'< 75% Traced Stream Water Fraction', 
                                     color='lightcoral', zorder=1, s=2)
    
    # Plot the "above" values with a dark red solid line
    # above_threshold = axs[i].plot(deltaQ_merged_threshold[column], linestyle='-', label=r'$\geq$ 75%', color='darkred', zorder=2)
    # below_threshold = axs[i].plot(deltaQ_merged[column], linestyle='-', label=r'< 75% Traced Stream Water Fraction', color='lightcoral', zorder=1)
    
    ## Get data for KGE above threshold
    # model_data = deltaQ_merged_threshold[column][start_index:]
    # dates = stream_iso_monthly_mean['Date'][start_index:]
    # valid_indices = ~np.isnan(obs_data) & ~np.isnan(model_data)
    
    # obs_data_ind = obs_data[valid_indices]
    # model_data_ind = model_data[valid_indices]
    
    # Calculate /KGE score for the current column
    deltaQ_nan = deltaQ_merged_threshold[column].dropna()
    kge, r, alpha, beta = he.evaluator(he.kge, deltaQ_merged_threshold[column], obs_data)
    # nse = he.evaluator(he.nse, model_data_ind, obs_data_ind)
    kge_value = kge[0]
    # kge_value = nse[0]
    kge_legend_labels.append(f'{column}: {kge_value:.2f}')
    
    axs[i].set_ylim(-13, -1)
    axs[i].grid(True)
    axs[i].text(0.02, 0.95, f"{site_names[i]}: KGE = {kge_value:.2f}", transform=axs[i].transAxes, va='top', ha='left', 
                fontsize=12, color='black')
    
    # Apply the date formatter to the x-axis
    axs[i].xaxis.set_major_formatter(date_formatter)
   
   # Set the major tick locator to a year interval
    axs[i].xaxis.set_major_locator(year_locator)
    start_date = pd.to_datetime('2015-12-01')  # Define your desired start date
    end_date = pd.to_datetime('2023-01-31')    # Define your desired end date
    axs[i].set_xlim(start_date, end_date)
    
    ## Legend
    # axs[0].legend(loc=(-0.1, 1.05),fontsize=12, ncol=3, handletextpad=0.1)  # Adjust the vertical spacing between the labels) #frameon=False,
    
## Add y label to middle of plots
fig.text(0.05, 0.5, '$\delta^{18}$O (‰)', va='center', ha='center', fontsize=12, rotation='vertical')

## Fix tick labels fontzise. Fix legend location
plt.setp([ax.get_xticklabels() for ax in axs], fontsize=12)
plt.setp([ax.get_yticklabels() for ax in axs], fontsize=12)

plt.subplots_adjust(wspace=0.1, hspace=0.15)
    

SAVE = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/Paper Draft/Figures/'
# plt.savefig(SAVE + 'Figure5_deltaQ.png', dpi=500)

## Save deltaQ threshold to dataframe for boxplot
column_names = deltaQ_merged.columns
df_deltaQ_threshold = pd.DataFrame(deltaQ_merged_threshold) 

# df_deltaQ_threshold.to_csv(path_deltaQ + 'Model_deltaQ_threshold.csv', index=False)

#%%

# Sort columns of MTT_merged in alphabetical order
MTT_merged = MTT_merged.sort_index(axis=1)

## Find the ranges in model MTT
# Create a dictionary to store the ranges for each column
time_change_ranges = {}

## Create a date range from January 2016 to December 2022
date_range = pd.date_range(start='2016-01-01', end='2022-12-31')

## Compare times!
# model_time = MTT_merged.values # /365 + 2018
obs_time   = date_range #stream_iso_monthly_mean['Date']

# Create a reference datetime object for the first day in 2018
first_day_2016 = pd.Timestamp('2016-01-01')
# Calculate the timedelta from the first day in 2018 for each datetime object
timedelta_from_2016 = obs_time - first_day_2016
# Convert timedelta values to days
timedelta_days = timedelta_from_2016 / np.timedelta64(1, 'D')

time_change_lst = []
# Iterate through the columns of MTT_merged
for i, column in enumerate(MTT_merged.columns):
    
    # Calculate the time change for the current column
    time_change = (timedelta_days.values - MTT_merged[column].values) /365 # Puts into years
    time_change_lst.append(time_change)

column_names = MTT_merged.columns
df_time_change = pd.DataFrame(time_change_lst).transpose()
df_time_change.columns = column_names

#%%
## Plot the MTT Time Series!!
site_names = ['a) HOPB','b) KING','c) LECO','d) MART','e) MCDI','f) SYCA']

# Sort columns of MTT_merged in alphabetical order
# MTT_merged = MTT_merged.sort_index(axis=1)

# ## Find the ranges in model MTT
# # Create a dictionary to store the ranges for each column
# time_change_ranges = {}

# ## Create a date range from January 2016 to December 2022
# date_range = pd.date_range(start='2016-01-01', end='2022-12-31')

# ## Compare times!
# # model_time = MTT_merged.values # /365 + 2018
# obs_time   = date_range #stream_iso_monthly_mean['Date']

# # Create a reference datetime object for the first day in 2018
# first_day_2016 = pd.Timestamp('2016-01-01')
# # Calculate the timedelta from the first day in 2018 for each datetime object
# timedelta_from_2016 = obs_time - first_day_2016
# # Convert timedelta values to days
# timedelta_days = timedelta_from_2016 / np.timedelta64(1, 'D')

# for i in MTT_merged:
    # time_change = timedelta_days.values - MTT_merged[i].values

# Create a 2x3 subplot
# fig, axs = plt.subplots(3, 2, sharex=True, figsize=(10, 12), dpi=500)
fig, axs = plt.subplots(6, 1, sharex=True, sharey=True, figsize=(8, 12), dpi=500)

# Flatten the axes array for easier iteration
axs = axs.flatten()

# Iterate through the columns of MTT_merged
for i, column in enumerate(df_time_change.columns):
    
    # Calculate the time change for the current column
    # time_change = timedelta_days.values - MTT_merged[column].values
    
    # For the "above" case
    values_above = tracer_threshold_above[column].values
    index_above = pd.Index(values_above).dropna()
    # deltaQ_above = deltaQ_merged[column].loc[index_above]
    
    ## Make below threshold values Nan
    # Get the DataFrame for the current column
    column_data = deltaQ_merged_threshold[column]

    # Create a mask where the index is not in the above index
    not_in_above_index_mask = ~column_data.index.isin(index_above)
    
    # Set the values in deltaQ_merged to NaN where the mask is True
    column_data.loc[not_in_above_index_mask] = pd.NA
    
    # Update the column in deltaQ_merged with the modified data
    deltaQ_merged_threshold[column] = column_data
    
    # Plot the time change on the respective subplot
    axs[i].plot(date_range, time_change/365)
    axs[i].text(0.01, 0.95, f"{site_names[i]}", transform=axs[i].transAxes, va='top', ha='left', fontsize=12, color='black')

    ## Add y label to middle of plots
    fig.text(0.05, 0.5, 'MTT (Years)', va='center', ha='center', fontsize=12, rotation='vertical')
    
    ## Fix limit on y axis
    ## Fix tick labels fontzise. Fix legend location
    plt.setp([ax.get_xticklabels() for ax in axs], fontsize=12)
    plt.setp([ax.get_yticklabels() for ax in axs], fontsize=12)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
        
    axs[i].grid(True)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    
    # Calculate the minimum and maximum values in time_change for the current column
    start_index = 600  # Change this value as needed
    time_change_after_index = time_change[start_index:]
    time_change = time_change_after_index[~np.isnan(time_change_after_index)]
    min_time_change = np.min(time_change)/365
    max_time_change = np.max(time_change)/365
    
    # Store the min and max values in the dictionary
    time_change_ranges[column] = (min_time_change, max_time_change)
    
    # Add the range information to the plot using text
    # range_text = f"Min: {min_time_change:.2f}, Max: {max_time_change:.2f}"
    # axs[i].text(0.41, 0.95, range_text, transform=axs[i].transAxes, va='top', ha='right', fontsize=12, color='black')
    
    start_date = pd.to_datetime('2015-12-01')  # Define your desired start date
    end_date = pd.to_datetime('2023-01-31')    # Define your desired end date
    axs[i].set_xlim(start_date, end_date)

SAVE = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/Paper Draft/Figures/'
# plt.savefig(SAVE + 'Figure4_MTT_TimeSeries.png', dpi=500,  bbox_inches='tight')

#%%
## Cross plot of above

# Create subplots
fig, axs = plt.subplots(2, 3, sharex=True, dpi=500, figsize=(12, 8))
axs = axs.ravel()
kge_legend_labels = []
for i, column in enumerate(common_columns):
    
    # Exclude NaN values from the data
    obs_data = obs_stream_Q[column][start_index:]
    model_data = deltaQ_merged[column][start_index:]
    valid_indices = ~np.isnan(obs_data) & ~np.isnan(model_data)
    obs_data = obs_data[valid_indices]
    model_data = model_data[valid_indices]
    
    # Calculate KGE score for the current column
    kge, _, _, _ = he.evaluator(he.kge, model_data, obs_data)
    kge_value = kge[0]
    kge_legend_labels.append(f'{column}: {kge_value:.2f}')

    # Plot cross plot of observed vs modeled stream isotopes
    axs[i].scatter(obs_data, model_data, label=f'{column}')
    # axs[i].set_xlabel('Observed $\delta^{18}$O (‰)')
    # axs[i].set_ylabel('Model $\delta^{18}$O (‰)')
    axs[i].set_xlim(-12, -3)  # Set x limits
    axs[i].set_ylim(-12, -3)  # Set y limits
    axs[i].set_title(f'{column}')
    axs[i].grid(True)
    plt.subplots_adjust(wspace=0.3, hspace=0.15)
    
    # Set labels on the outer edges of the left two plots and bottom plots
    if i == 0 or i == 3:
        axs[i].set_ylabel('Modeled $\delta^{18}$O (‰)')
    if i == 3 or i == 4 or i == 5:
        axs[i].set_xlabel('Observed $\delta^{18}$O (‰)')

# Create a legend for KGE values
fig.legend(kge_legend_labels, title='KGE Score', loc='upper right', bbox_to_anchor=(1.1, 0.5))

#%%
# Create a date range from January 2016 to December 2022
date_range = pd.date_range(start='2016-01-01', end='2022-12-31')

## Compare times!
model_time = (Time_df[stream_id].values) # /365 + 2018
obs_time   = date_range #stream_iso_monthly_mean['Date']

# Create a reference datetime object for the first day in 2018
first_day_2016 = pd.Timestamp('2016-01-01')
# Calculate the timedelta from the first day in 2018 for each datetime object
timedelta_from_2016 = obs_time - first_day_2016
# Convert timedelta values to days
timedelta_days = timedelta_from_2016 / np.timedelta64(1, 'D')
time_change = timedelta_days.values - model_time


plt.figure(figsize=(10, 6), dpi=500)
# plt.scatter(stream_iso_monthly_mean['Date'], Time_df[stream_id]/365 + 2018, color='blue')
# plt.ylabel('Cumulative weighted time (Days)')
# plt.xlabel('Mean Transport Time')
# plt.title('Mean Transport Time Over Time')
# plt.legend()

# plt.plot(stream_iso_monthly_mean['Date'], time_change/365, color='blue')
plt.plot(date_range, time_change/365, color='blue')
plt.ylabel('MTT (years)')
plt.grid(True)


