#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:12:13 2024

@author: butl842
"""

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
import calendar
import glob 

name_id = "SYCA"  # Prompt user for a name id

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
path_Fyw          = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/Fyw_TimeSeries/'


df_nan = pd.read_csv(path + f'{name_id}_TracerFiles.csv')
df = df_nan[df_nan.iloc[:,:] != -9999] # Exclude rows with -9999 valuesx
df['Date'] = pd.to_datetime(df['Date'])

## Convert streamflow m3/s to hourly values because that is date time format
columns_to_multiply = df.columns[1:]
df[columns_to_multiply] = ((df[columns_to_multiply]*3600) / area) * 1000 # Converts m3/hr to mm/hr

# Set 'Date' as the index
df.set_index('Date', inplace=True)

## Group streamflow by month and sum each column to get monthly traced streamflow
df_monthly_sum = df.resample('M').sum() # mm/hr to mm/month
df_daily_sum   = df.resample('D').sum() # m3/hr to m3/month

# Reset the index. This gives me a monthly sum of each tracer and the total streamflow
df_monthly_sum.reset_index(inplace=True)
df_daily_sum.reset_index(inplace=True)

df_total = df_daily_sum[['Date', 'TotalStreamflow']].copy() # Make new df of Total Streamflow per month
df_trace = df_daily_sum.drop(columns=['TotalStreamflow'])  # Make new df of Traced Streamflow per month
df_trace = df_trace.drop(columns=['Date'])

col_name = df_trace.columns # col names

start_day = []
for i in np.arange(len(col_name)):
    cur_year  = int(col_name[i].split('_')[1])
    cur_month = int(col_name[i].split('_')[2])
    date_time = datetime(cur_year, cur_month, 1) - datetime(2016,1,1) # Create start day for each month
    start_day.append(date_time.days)

start_day = np.array(start_day) #convert list to array 
Fyw_days  = 90
Fyw_frac  = []
## Loop through df
for i in np.arange(len(df_trace)):
    tracer_i = df_trace.iloc[i]
    cur_days_past = i - start_day
    ## Take days less than 9- days and want above 0 days so no negatives divided by the sum of that row
    cur_fraction  =  np.sum(tracer_i[ (cur_days_past<=Fyw_days) & (cur_days_past >= 0) ]) / np.sum(tracer_i)
    Fyw_frac.append(cur_fraction)
    
Fyw_df = pd.DataFrame(Fyw_frac, columns=[name_id])

# Fyw_df.to_csv(path_Fyw + f"{name_id}_Fyw.csv", index=False)

#%%
## Merge all files together
Fyw_files = glob.glob(path_Fyw + "/*.csv")
Fyw_merge_lst = [] 
Column_names  = []
# Read and append CDF_files files
for file_path in Fyw_files:
    site = file_path.split('/')[-1][:4]
    Column_names.append(site)
    
    df_Fyw = pd.read_csv(file_path)
    Fyw_merge_lst.append(df_Fyw)
    
Fyw_merged = pd.concat(Fyw_merge_lst, axis=1)
Fyw_merged.columns = Column_names
# Sort columns in alphabetical order
Fyw_merged = Fyw_merged.sort_index(axis=1)

tracer_range_2021 = Fyw_merged.iloc[-366:]
range_2021 = tracer_range_2021.agg([min, max]) * 100
mean_2021  = tracer_range_2021.mean() * 100

#%%
path_threshold = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/MassSum/Threshold/'

## Read in threshold data
tracer_threshold_above = pd.read_csv(path_threshold + 'Tracer_index_threshold_above.csv')
tracer_threshold_below = pd.read_csv(path_threshold + 'Tracer_index_threshold_below.csv')
TotalQ_threshold_above = pd.read_csv(path_threshold + 'TotalQ_index_threshold_above.csv')
TotalQ_threshold_below = pd.read_csv(path_threshold + 'TotalQ_index_threshold_below.csv')

## Plot the MTT Time Series!!
site_names = ['a) HOPB','b) KING','c) LECO','d) MART','e) MCDI','f) SYCA']

## Create a date range from January 2016 to December 2022
date_range = pd.date_range(start='2016-01-01', end='2022-12-31')

fig, axs = plt.subplots(6, 1, sharex=True, sharey=False, figsize=(8, 12), dpi=500)

# Flatten the axes array for easier iteration
axs = axs.flatten()
   
Fyw_time_threshold = Fyw_merged.copy() # Mkae copy to acess both 
# Iterate through the columns of MTT_merged
for i, column in enumerate(Fyw_merged.columns):
    
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
    column_data = Fyw_time_threshold[column]

    # Create a mask where the index is not in the above index
    not_in_above_index_mask = ~column_data.index.isin(common_indexes)
    
    # Set the values in deltaQ_merged to NaN where the mask is True
    column_data.loc[not_in_above_index_mask] = pd.NA
    
    # Update the column in deltaQ_merged with the modified data
    Fyw_time_threshold[column] = column_data
    
    # Convert to percent
    Fyw_days_thres = Fyw_time_threshold * 100
    Fyw_merged_percent = Fyw_merged * 100
    
    axs[i].scatter(date_range, Fyw_days_thres[column], color='darkred', zorder=2, label=r'$\geq$ 75%', s=2)
    axs[i].scatter(date_range, Fyw_merged_percent[column], color='lightcoral', zorder=1, label=r'< 75% Traced Stream Water Fraction', s=2)
    
    axs[i].text(0.02, 0.95, f"{site_names[i]}", transform=axs[i].transAxes, va='top', ha='left', fontsize=12, color='black')

    ## Add y label to middle of plots
    fig.text(0.03, 0.5, '$\it{F_{yw}}$ (%)', va='center', ha='center', fontsize=12, rotation='vertical')
    axs[i].set_ylim(-5, 125)
    # Set the same number of y ticks for each subplot
    axs[i].set_yticks([0, 25, 50, 75, 100])
    
    ## Fix limit on y axis
    ## Fix tick labels fontzise. Fix legend location
    plt.setp([ax.get_xticklabels() for ax in axs], fontsize=12)
    plt.setp([ax.get_yticklabels() for ax in axs], fontsize=12)
    
    axs[i].grid(True)
    plt.subplots_adjust(wspace=0.1, hspace=0.15)
    
    start_date = pd.to_datetime('2015-12-01')  # Define your desired start date
    end_date = pd.to_datetime('2023-01-31')    # Define your desired end date
    axs[i].set_xlim(start_date, end_date)
    # axs[0].legend(loc=(0.1, 1.05),fontsize=12, ncol=2)  # Adjust the vertical spacing between the labels) #frameon=False,

SAVE = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/Paper Draft/Figures/'
path_MTT    = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/'
plt.savefig(SAVE + 'Figure5_Fyw_TimeSeries.png', dpi=500,  bbox_inches='tight')

## Save MTT threshold to dataframe for boxplot
column_names = Fyw_days_thres.columns
df_Fyw_threshold = pd.DataFrame(Fyw_days_thres) 

# df_Fyw_threshold.to_csv(path_MTT + 'Model_Fyw_threshold.csv', index=False)
