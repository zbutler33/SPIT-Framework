#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:18:40 2024

@author: butl842
"""

import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator
import hydroeval as he
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob 

path_MTT    = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/'
path_threshold = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/MassSum/Threshold/'
path_Fyw          = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/Fyw_TimeSeries/'

## Read in threshold data
tracer_threshold_above = pd.read_csv(path_threshold + 'Tracer_index_threshold_above.csv')
tracer_threshold_below = pd.read_csv(path_threshold + 'Tracer_index_threshold_below.csv')
TotalQ_threshold_above = pd.read_csv(path_threshold + 'TotalQ_index_threshold_above.csv')
TotalQ_threshold_below = pd.read_csv(path_threshold + 'TotalQ_index_threshold_below.csv')
MTT_time = pd.read_csv(path_MTT + 'MTT_TimeSeries.csv')

tracer_range_2021 = MTT_time.iloc[-366:]
range_2021 = tracer_range_2021.agg([min, max]) * 365.5
mean_2021  = tracer_range_2021.mean() * 365.5

#%%
## Plot the MTT Time Series!!
site_names = ['a) HOPB','b) KING','c) LECO','d) MART','e) MCDI','f) SYCA']

# Sort columns of MTT_merged in alphabetical order
# MTT_merged = MTT_merged.sort_index(axis=1)

# ## Find the ranges in model MTT
# # Create a dictionary to store the ranges for each column
# time_change_ranges = {}

## Create a date range from January 2016 to December 2022
date_range = pd.date_range(start='2016-01-01', end='2022-12-31')

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
fig, axs = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(8, 12), dpi=500)

# Flatten the axes array for easier iteration
axs = axs.flatten()

MTT_time_threshold = MTT_time.copy() # Mkae copy to acess both
# Iterate through the columns of MTT_merged
for i, column in enumerate(MTT_time.columns):
    
    # For the "above" case
    values_above = tracer_threshold_above[column].values
    index_above = pd.Index(values_above).dropna()
    
    # For the "above" case using TotalQ_threshold_above
    values_above_TotalQ = TotalQ_threshold_above[column].values
    index_above_TotalQ = pd.Index(values_above_TotalQ).dropna()
    
    # Combine both indexes
    common_indexes = index_above.intersection(index_above_TotalQ)
    
    ## Make below threshold values Nan
    # Get the DataFrame for the current column
    column_data = MTT_time_threshold[column]

    # Create a mask where the index is not in the above index
    not_in_above_index_mask = ~column_data.index.isin(common_indexes)
    
    # Set the values in deltaQ_merged to NaN where the mask is True
    column_data.loc[not_in_above_index_mask] = pd.NA
    
    # Update the column in deltaQ_merged with the modified data
    MTT_time_threshold[column] = column_data
    
    # Convert to days
    MTT_days_thres = MTT_time_threshold * 365.5
    MTT_days = MTT_time * 365.5
    
    # Plot the time change on the respective subplot
    # axs[i].plot(date_range, time_change/365)
    # axs[i].plot(date_range, MTT_days_thres[column], color='darkred', zorder=2, label=r'$\geq$ 75%')
    # axs[i].plot(date_range, MTT_time[column], color='lightcoral', zorder=1, label=r'< 75% Traced Stream Water Fraction')
    
    axs[i].scatter(date_range, MTT_days_thres[column], color='darkred', zorder=2, label=r'$\geq$ 75%', s=2)
    axs[i].scatter(date_range, MTT_days[column], color='lightcoral', zorder=1, label=r'< 75% Traced Stream Water Fraction', s=2)
    # axs[i].set_yticks(range(0, 1461, 365))
    # Set the same number of y ticks for each subplot
    axs[0].set_yticks([0, 300, 600])
    axs[1].set_yticks([0, 300, 600, 900])
    axs[2].set_yticks([0, 300, 600])
    axs[3].set_yticks([0, 300, 600])
    axs[4].set_yticks([0, 300, 600, 900])
    axs[5].set_yticks([0, 300, 600, 900, 1200, 1500])
    
    axs[i].text(0.02, 0.95, f"{site_names[i]}", transform=axs[i].transAxes, va='top', ha='left', fontsize=12, color='black')

    ## Add y label to middle of plots
    fig.text(0.03, 0.5, 'MTT (Days)', va='center', ha='center', fontsize=12, rotation='vertical')
    
    ## Fix limit on y axis
    ## Fix tick labels fontzise. Fix legend location
    plt.setp([ax.get_xticklabels() for ax in axs], fontsize=12)
    plt.setp([ax.get_yticklabels() for ax in axs], fontsize=12)
    
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
        
    axs[i].grid(True)
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    
    start_date = pd.to_datetime('2015-12-01')  # Define your desired start date
    end_date = pd.to_datetime('2023-01-31')    # Define your desired end date
    axs[i].set_xlim(start_date, end_date)
    # axs[0].legend(loc=(0.1, 1.05),fontsize=12, ncol=2)  # Adjust the vertical spacing between the labels) #frameon=False,

SAVE = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/Paper Draft/Figures/'
plt.savefig(SAVE + 'Figure4_MTT_TimeSeries.png', dpi=500, bbox_inches='tight')

## Save MTT threshold to dataframe for boxplot
column_names = MTT_time.columns
df_MTT_threshold = pd.DataFrame(MTT_time_threshold) 

# df_MTT_threshold.to_csv(path_MTT + 'Model_MTT_threshold.csv', index=False)
