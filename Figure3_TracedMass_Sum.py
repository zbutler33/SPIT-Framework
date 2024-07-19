#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:49:56 2024

@author: butl842

This script finds the percent of mass traced for the forward and backward TTD.
The forward TTD is the fraction of traced streamflow to the incoming precipitation.
The backward TTD is the fraction of traced streamflow to the total streamflow.
    - Each of the X of 72 TTDs is plotted and saved, as well as a GIF of all 72 through time.

Figure 3: From the Backward TTD, a tracer time series is plotted with the precipitation as bars.
    - The Percent of Mass tracked for the avg of the X of 72 TTDs is then plotted.
        - Shows when in time we have a high percent of mass part of the TTD
        - Can lead into if we want to use a threshold for the MTT calculations...
    - Also plots the normalized PDF/CDF 
    
- See TTD_Stepback.py for each plot of the 72 tracer tracking and GIF.
- See TTD_Normalized for normalized pdf and cdf as well as season information. 
    
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import calendar

name_id = "KING"  # Prompt user for a name id

if name_id == "HOPB":
    area = 13.390536106870169 * 1000000 # (1000*1000) #convert km2 to m2. # Gridded watershed area
elif name_id == "KING":
    area = 12.527744014923439 * 1000000 # NEON 13.2795
elif name_id == "LECO":
    area = 9.169550503924794 * 1000000 # 9.5553
elif name_id == "MART":
    area = 7.78861648634644 * 1000000  # 7.78861648634644 convert km2 to m2
elif name_id == "MCDI":
    area = 22.32275447072249 * 1000000 
elif name_id == "SYCA":
    area = 264.8246387688475 * 1000000

# elif name_id == "Yakima":
#     area = 
    
path              = f'/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/{name_id}/'
path_tracer       = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/TracerStreamflow/Daily/'
save_path_backward = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/MassSum/'
save_path_forward  = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/ForwardTTD/MassSum/'
precip_path       = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/Precipitation/'
path_backward = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/'

df_Total = pd.read_csv(path_backward + f'TotalQ.csv')

df_precip = pd.read_csv(path + f'{name_id}_Weighted_MonthlyPrecip.csv')
# df_precip = pd.read_csv(path + f'{name_id}_Weighted_MonthlyPrecip_First.csv')
# df_precip = pd.read_csv(path + f'{name_id}_Weighted_MonthlyPrecip_Hourly.csv')
# df_precip_hour = pd.read_csv(path + f'{name_id}_Weighted_DailyPrecip_Hourly.csv')

# Convert units
# df_precip['MonthlyPrecip (m2/mm)'] = df_precip['MonthlyPrecip'] #* area # convert mm to m2*mm
# df_precip['MonthlyPrecip m3'] = df_precip['MonthlyPrecip (m2/mm)'] * 0.001   # convert m2*mm to m3

# Check that each value is each month of precipitation summed over the watershed!
df_inputP = df_precip['MonthlyPrecip_depth']
transposed_df_inputP = df_inputP.to_frame().T #transpose to make division over each column

# df_precip_hour['Date'] = pd.to_datetime(df_precip_hour['Date'], format='%Y%m%d')
# df_precip_hour.set_index('Date', inplace=True)
# df_Precip_day_month   = df_precip_hour.resample('M').sum() # m3/hr to m3/month
# df_inputP_hour = df_Precip_day_month['DailyPrecip_depth']
# transposed_df_inputP_hour = df_inputP_hour.to_frame().T

df_nan = pd.read_csv(path + f'{name_id}_TracerFiles.csv')
df = df_nan[df_nan.iloc[:,:] != -9999] # Exclude rows with -9999 valuesx
df['Date'] = pd.to_datetime(df['Date'])

## Convert streamflow m3/s to hourly values because that is date time format
columns_to_multiply = df.columns[1:]
# df[columns_to_multiply] = df[columns_to_multiply] * 3600 # Convert m3/s to m3/hr
df[columns_to_multiply] = ((df[columns_to_multiply]*3600) / area) * 1000 # Converts m3/hr to mm/hr

# Divide by areas to make into mm. For plotting in Figure 7. 
specific_discharge  = (df[columns_to_multiply] / area)  * 1000
# per unit areas

# Set 'Date' as the index
df.set_index('Date', inplace=True)

## Group streamflow by month and sum each column to get monthly traced streamflow
df_monthly_sum = df.resample('M').sum() # mm/hr to mm/month
df_daily_sum   = df.resample('D').sum() # m3/hr to m3/month


## are times slightly different
## could be missing first or last day of month

# Reset the index. This gives me a monthly sum of each tracer and the total streamflow
df_monthly_sum.reset_index(inplace=True)
df_daily_sum.reset_index(inplace=True)
# df_total = df_monthly_sum[['Date', 'TotalStreamflow']].copy() # Make new df of Total Streamflow per month
# df_trace = df_monthly_sum.drop(columns=['TotalStreamflow'])  # Make new df of Traced Streamflow per month
# df_trace = df_trace.drop(columns=['Date'])

df_total = df_daily_sum[['Date', 'TotalStreamflow']].copy() # Make new df of Total Streamflow per month
df_trace = df_daily_sum.drop(columns=['TotalStreamflow'])  # Make new df of Traced Streamflow per month
df_trace = df_trace.drop(columns=['Date'])

df_trace_month = df_monthly_sum.drop(columns=['TotalStreamflow'])  # Make new df of Traced Streamflow per month
df_trace_month = df_trace_month.drop(columns=['Date'])

## Make into csv for check
# df_total.to_csv(path + f"{name_id}_TotalStreamflow.csv", index=False)
# df_trace.to_csv(path + f"{name_id}_TracedStreamflow.csv", index=False)

## Accounts for other 0's in dataframes
def roll_up_on_zero(column):
    first_non_zero_index = next((i for i, x in enumerate(column) if x != 0), None)
    if first_non_zero_index is not None:
        non_zero_values = [column[i] for i in range(first_non_zero_index, len(column))]
        return non_zero_values + [0] * (len(column) - len(non_zero_values))
    else:
        return column
    
#%%
## Make the Backward TTD
df_output = df_total['TotalStreamflow']
transposed_df_output = df_output.to_frame().T #transpose to make division over each column
# transposed_df_output.columns = df_trace.columns[:] # Make column names the same for loop below
df_drop = df_trace.iloc[:, :]
tracer_transpose = df_drop.T 

flipped_tracer_transpose = tracer_transpose[::-1]
# df_trace_backward = flipped_tracer_transpose.iloc[:, :].apply(roll_up_on_zero, axis=0)

# Apply the function to each column
df_trace_backward = flipped_tracer_transpose.apply(roll_up_on_zero, axis=0)

# df_trace_backward.columns = df_trace.columns[:] # Make column names the same for loop below
# df_trace_backward.to_csv(save_path + "Backward_TracedStreamflow.csv", index=False)

# transposed_df_output.to_csv(save_path + "Backward_TotalStreamflow.csv", index=False)

## Divide each column by single value of precipitation
df_backward = df_trace_backward.copy()
for column_back in df_trace_backward.columns[:]:
    df_backward[column_back] = df_trace_backward[column_back] / transposed_df_output[column_back].values

# Compute the sum of each column
sum_backward = df_backward.sum(axis=0)
df_sum_backward = pd.DataFrame(sum_backward, columns=[name_id])
# df_sum_backward.to_csv(save_path_backward + f'{name_id}_MassSum.csv', index=False)

#%%
# df_trace_forward = df_trace_month.iloc[:, :].apply(roll_up_on_zero, axis=0)
# df_trace_forward.to_csv(path + "Forward_TracedStreamflow.csv", index=False)
# # transposed_df_inputP.to_csv(path + "Forward_InputPrecipitation.csv", index=False)

# ## Divide each column by single value of precipitation
# df_forward = df_trace_forward.copy() # copy so dataframe is set up the same. Overwrite below with calculation

# # Ensure that transposed_df_inputP has the same column names as df_trace_forward
# transposed_df_inputP.columns = df_trace_month.columns[:] # Make column names the same for loop below
# for column_for in df_trace_forward.columns[:]:
#     # Divide each tracer concentration by the input precipitation
#     tracer_forward = df_trace_forward[column_for]
#     precipitation_forward = transposed_df_inputP[column_for].values

#     ## Fracton of stream water from precipitation
#     df_forward[column_for] = df_trace_forward[column_for] / transposed_df_inputP[column_for].values

#######

# df_trace_forward_T = df_trace_forward.copy()
# transposed_df_inputP_hour.columns = df_trace.columns[:]
# for column_for in df_trace_forward.columns[:]:
    
#     tracer_forward = df_trace_forward[column_for]
#     precipitation_forward_hour = transposed_df_inputP_hour[column_for].values
    
#     # Divide each tracer concentration by the input precipitation
#     df_trace_forward_T[column_for] = df_trace_forward[column_for] / transposed_df_inputP_hour[column_for].values
    
########

# Compute the sum of each column
# sum_forward = df_forward.sum(axis=0)
# # sum_forward_hour = df_trace_forward_T.sum(axis=0)

# # Append the sums as a new row to the DataFrame
# df_forward_sum = df_forward.append(sum_forward, ignore_index=True)

# df_sum_forward = pd.DataFrame(sum_forward, columns=[name_id])
# df_sum_forward.to_csv(save_path_forward + f'{name_id}_MassSum.csv', index=False)

#%%
import glob
## Merge backward TTD together to plot
Sum_files    = glob.glob(save_path_backward + "/*.csv") # Load all files
Sum_files_for    = glob.glob(save_path_forward + "/*.csv")
# Initialize empty lists to store DataFrames
tracer_dfs = []
tracer_dfs_for = []

# Read and append CDF_files files
for file_path in Sum_files:
    df_sum_tracers = pd.read_csv(file_path)
    tracer_dfs.append(df_sum_tracers)
    
## Merge  files into one DataFrame
tracer_merged = pd.concat(tracer_dfs, axis=1)

for file_path in Sum_files_for:
    df_sum_tracers_for = pd.read_csv(file_path)
    tracer_dfs_for.append(df_sum_tracers_for)
    
## Merge  files into one DataFrame
tracer_merged_for = pd.concat(tracer_dfs_for, axis=1)

#%%
## Create a threshold to find the Traced Stream Water Fraction above a certai sum

save_threshold = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/MassSum/Threshold/'

threshold = 0.75  # Example threshold value of 80%

# Dictionary to store indices for each column
Threshold_lst = []
threshold_above = {}
threshold_below = {}
indices_above_threshold = []
indices_below_threshold = []

# Iterate through each column in tracer_merged
for i, column in enumerate(tracer_merged.columns):
    
    # Create a boolean mask for the current column where the values are above the threshold
    mass_above = tracer_merged[column] >= threshold
    mass_below = tracer_merged[column] < threshold
    
    # Use boolean indexing to create a new Series where values below the threshold are set to NaN
    threshold_mass_above = tracer_merged[column].where(mass_above, np.nan)
    threshold_mass_below = tracer_merged[column].where(mass_below, np.nan)
    
    # Store the filtered column in the dictionary
    threshold_above[column] = threshold_mass_above
    threshold_below[column] = threshold_mass_below
    
    # Use boolean masks to filter the indices and convert them to lists
    indices_above = list(tracer_merged.index[mass_above])
    indices_below = list(tracer_merged.index[mass_below])
    
    # Append the lists to the respective lists
    indices_above_threshold.append(indices_above)
    indices_below_threshold.append(indices_below)

# Convert the dictionary to a DataFrame
tracer_threshold = pd.DataFrame(threshold_above)
tracer_threshold_below = pd.DataFrame(threshold_below)
# tracer_threshold = tracer_threshold.sort_index(axis=1)
# tracer_threshold_below = tracer_threshold_below.sort_index(axis=1)
column_names = tracer_threshold.columns

df_indices_above_threshold = pd.DataFrame(indices_above_threshold).transpose()
df_indices_below_threshold = pd.DataFrame(indices_below_threshold).transpose()
df_indices_above_threshold.columns = column_names
df_indices_below_threshold.columns = column_names

df_indices_above_threshold = df_indices_above_threshold.sort_index(axis=1)
df_indices_below_threshold = df_indices_below_threshold.sort_index(axis=1)
tracer_threshold_above = tracer_threshold.sort_index(axis=1)
tracer_threshold_below = tracer_threshold_below.sort_index(axis=1)

## Get range and median for last year of data
# Select the last 366 indexes
tracer_range_2021 = tracer_merged.iloc[-366:]
range_2021 = tracer_range_2021.agg([min, max])
mean_2021  = tracer_range_2021.mean()

# tracer_range_below_2021 = tracer_threshold_below.iloc[-366:]
# range_below_2021 = tracer_range_below_2021.agg([min, max])
# mean_below_2021  = tracer_range_below_2021.mean()
tracer_threshold_above.to_csv(save_threshold + 'Tracer_threshold_above.csv', index=False)
tracer_threshold_below.to_csv(save_threshold + 'Tracer_threshold_below.csv', index=False)
df_indices_above_threshold.to_csv(save_threshold + 'Tracer_index_threshold_above.csv', index=False)
df_indices_below_threshold.to_csv(save_threshold + 'Tracer_index_threshold_below.csv', index=False)

#%%

SAVE = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/Paper Draft/Figures/'
path_threshold = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/MassSum/Threshold/'

site_names = ['a) HOPB','b) KING','c) LECO','d) MART','e) MCDI','f) SYCA']

## Read in threshold data
tracer_threshold_above_index = pd.read_csv(path_threshold + 'Tracer_index_threshold_above.csv')
tracer_threshold_below_index = pd.read_csv(path_threshold + 'Tracer_index_threshold_below.csv')
TotalQ_threshold_above = pd.read_csv(path_threshold + 'TotalQ_index_threshold_above.csv')
TotalQ_threshold_below = pd.read_csv(path_threshold + 'TotalQ_index_threshold_below.csv')

# Sort columns of MTT_merged in alphabetical order
tracer_merged = tracer_merged.sort_index(axis=1)
tracer_merged_for = tracer_merged_for.sort_index(axis=1)

fig, axs = plt.subplots(6, 1, sharex=True, sharey=True, dpi=500, figsize=(8, 12))
axs = axs.ravel()

threshold = 75
# Define the start index for excluding the first year of data
start_index = 0  # Assuming monthly data
tracer_merged_threshold = tracer_merged.copy() # Mkae copy to acess both
days_above_X_lst = []

# Iterate through the columns of MTT_merged
for i, column in enumerate(tracer_merged.columns):

    # Find Max values
    max_sum = tracer_merged[column].max() * 100
    
    ## Number of days above X%
    days_above_X = (tracer_merged[column] >= 0.995).sum()
    days_above_X_lst.append(days_above_X)
    
    # axs[i].plot(df_total['Date'], tracer_merged[column] * 100, label=r'Backward = $\frac{Traced\ Streamflow}{Total\ Streamflow}$', color='navy')
    
    # For the "above" case
    values_above = tracer_threshold_above_index[column].values
    index_above = pd.Index(values_above).dropna()
    # deltaQ_above = deltaQ_merged[column].loc[index_above]
    
    # For the "above" case using TotalQ_threshold_above
    values_above_TotalQ = TotalQ_threshold_above[column].values
    index_above_TotalQ = pd.Index(values_above_TotalQ).dropna()
    
    # Combine both indexes
    common_indexes = index_above.intersection(index_above_TotalQ)
    
    ## Make below threshold values Nan
    # Get the DataFrame for the current column
    column_data = tracer_merged_threshold[column]

    # Create a mask where the index is not in the above index
    not_in_above_index_mask = ~column_data.index.isin(common_indexes)
    
    # Set the values in deltaQ_merged to NaN where the mask is True
    column_data.loc[not_in_above_index_mask] = pd.NA
ocl    
    # # Update the column in deltaQ_merged with the modified data
    tracer_merged_threshold[column] = column_data
    
    ## Plot threshold solid and dotted
    # above_threshold = axs[i].plot(df_total['Date'], tracer_merged_threshold[column] * 100, label=r'$\geq$ 75%', 
    #                               color='darkred', zorder=2) # $\frac{Traced\ Streamflow}{Total\ Streamflow}$
    # below_threshold = axs[i].plot(df_total['Date'], tracer_merged[column] * 100, label=r'< 75% Traced Stream Water Fraction', 
    #             color='lightcoral', zorder=1)
    
    # Count how many days are above 0.75
    last_366_SYCA = tracer_merged_threshold['SYCA'].iloc[-366:]
    count_above_075 = (last_366_SYCA >= 0.75).sum()
    
    # Plot scatter dots with customized sizes
    scatter_above = axs[i].scatter(df_total['Date'], tracer_merged_threshold[column] * 100, label=r'$\geq$ 75%', 
                                   color='darkred', zorder=2, s=2) # Adjust the marker size here
    scatter_below = axs[i].scatter(df_total['Date'], tracer_merged[column] * 100, label=r'< 75% Traced Stream Water Fraction', 
                                   color='lightcoral', zorder=1, s=2) # Adjust the marker size here
    
    # axs[i].plot(df_total['Date'], tracer_merged_for[column] * 100, label=r'Backward = $\frac{Traced\ Streamflow}{Precipitation\}$', color='crimson')
    axs[i].text(0.02, 0.95, f"{site_names[i]}", transform=axs[i].transAxes, va='top', ha='left', fontsize=12, color='black')
    axs[i].grid(True)
    # axs[5].set_xlabel('Months', fontsize=12)
    fig.text(0.05, 0.5, 'Traced Stream Water Fraction', va='center', ha='center', fontsize=12, rotation='vertical')
    # axs[i].set_xlim(-1,84,1)
    axs[i].set_ylim(-1,120,1)
    
    # Format the range information as a string
    # range_text = f"Max = {max_sum:.1f}%"
    # axs[i].text(0.32, 0.95, range_text, transform=axs[i].transAxes, va='top', ha='right', fontsize=12, color='black')

    ## Fix x limits, y scales to fit a-f, max values next to site name
    start_date = df_total['Date'].iloc[0]
    end_date = df_total['Date'].iloc[-1]
    date_range = end_date - start_date
    extra_space = date_range * 0.014  # Adjust this factor as per your preference
    axs[i].set_xlim(start_date - extra_space, end_date + extra_space)
    
    # axs[0].legend(loc=(0.2, 1.05),fontsize=12, ncol=2) #frameon=False,
# Calculate the average number of days above 0.95 across all columns
average_days_above_X = sum(days_above_X_lst) / len(days_above_X_lst)

# plt.savefig(SAVE + 'Figure3_SummedMass.png', dpi=500, bbox_inches='tight')
