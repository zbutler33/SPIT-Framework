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

name_id = "HOPB"  # Prompt user for a name id

if name_id == "HOPB":
    area = 13.390536106870169 * 1000000 # (1000*1000) #convert km2 to m2. # Gridded watershed area
elif name_id == "KING":
    area = 12.527744014923439 * 1000000 # NEON 13.2795
elif name_id == "LECO":
    area = 9.169550503924794 * 1000000 # 9.5553
elif name_id == "MART":
    area = 7.78861648634644 * 1000000  #convert km2 to m2
elif name_id == "MCDI":
    area = 22.32275447072249 * 1000000 
elif name_id == "SYCA":
    area = 264.8246387688475 * 1000000
    

path              = f'/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/{name_id}/'
path_tracer       = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/TracerStreamflow/Daily/'
precip_path       = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/Precipitation/'
precip_path_units = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/Precipitation/Units/'

# save_tracer       = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/'
# save_total        = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/TotalStreamflow/'
save_path_forward = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/ForwardTTD/'
save_path_backward = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/TracerFiles/BackwardTTD/'

df_nan = pd.read_csv(path + f'{name_id}_TracerFiles.csv')
df = df_nan[df_nan.iloc[:,:] != -9999] # Exclude rows with -9999 valuesx
df['Date'] = pd.to_datetime(df['Date'])

df_precip = pd.read_csv(path + f'{name_id}_Weighted_MonthlyPrecip.csv')

## Convert units. 
# df_precip['MonthlyPrecip (m2/mm)'] = df_precip['MonthlyPrecip'] * area # convert mm to m2*mm
# df_precip['MonthlyPrecip m3'] = df_precip['MonthlyPrecip (m2/mm)'] * 0.001   # convert m2*mm to m3

## Load precipitation values in mm per unit area
# df_inputP = df_precip['MonthlyPrecip m3'] # m3 per month
df_inputP = df_precip['MonthlyPrecip_depth'] # mm per month
transposed_df_inputP = df_inputP.to_frame().T #transpose to make division over each column
df_precip.to_csv(path + f'{name_id}_MonthlyPrecip_Units.csv') # Save precip into new dataframe

## Convert streamflow m3/s to hourly values because that is the model output time
columns_to_multiply = df.columns[1:]

## New Conversion test to mm of streamflow
df[columns_to_multiply] = ((df[columns_to_multiply]*3600) / area) * 1000 # Converts m3/hr to mm/hr
## ((m3/hr) / m2) = m/hr. (m/hr * 1000mm) = mm/hr 
# df[columns_to_multiply] = df[columns_to_multiply] * 3600 # Convert m3/s to m3/hr OLD!!

# Set 'Date' as the index
df.set_index('Date', inplace=True)

## Group streamflow by month and sum each column to get monthly traced streamflow
df_monthly_sum = df.resample('M').sum() # m3/hr or mm/hr to m3/month or mm/month
df_daily_sum   = df.resample('D').sum() # m3/hr to m3/month

# Reset the index. This gives me a monthly sum of each tracer and the total streamflow
df_monthly_sum.reset_index(inplace=True)
df_total = df_monthly_sum[['Date', 'TotalStreamflow']].copy() # Make new df of Total Streamflow per month
df_trace = df_monthly_sum.drop(columns=['TotalStreamflow'])  # Make new df of Traced Streamflow per month
# df_trace = df_daily_sum.drop(columns=['TotalStreamflow'])

specific_discharge  = (df_monthly_sum['TotalStreamflow'] / area)  * 1000
df_specific = pd.concat([df_total['Date'], specific_discharge], axis=1)
df_specific.columns = ['Date', 'SpecificDischarge']  # Rename the column

## Make into csv for check
# df_total.to_csv(path + f"{name_id}_TotalStreamflow.csv", index=False)
# df_trace.to_csv(path + f"{name_id}_TracedStreamflow.csv", index=False)
# transposed_df_inputP.to_csv(path + "PrecipitationInput.csv", index=False)
# df_specific.to_csv(path + f"{name_id}_SpecificDischarge.csv", index=False)

df_trace = df_trace.drop(columns=['Date']) 

transposed_df_inputP.columns = df_trace.columns[:] # Make column names the same for loop below

## Roll up 0's so each column starts at same index 
# def roll_up_on_zero(column):
#     zero_indices = column.index[column == 0].tolist()
#     for idx in zero_indices:
#         column = np.roll(column, -1)
#         column[-1] = 0
#     return column

## Accounts for other 0's in dataframes
def roll_up_on_zero(column):
    first_non_zero_index = next((i for i, x in enumerate(column) if x != 0), None)
    if first_non_zero_index is not None:
        non_zero_values = [column[i] for i in range(first_non_zero_index, len(column))]
        return non_zero_values + [0] * (len(column) - len(non_zero_values))
    else:
        return column
    
#%%
df_trace_forward = df_trace.iloc[:, :].apply(roll_up_on_zero, axis=0)
# df_trace_forward.to_csv(path + "Forward_TracedStreamflow.csv", index=False)
# transposed_df_inputP.to_csv(path + "Forward_InputPrecipitation.csv", index=False)

## Divide each column by single value of precipitation
df_forward = df_trace_forward.copy() # copy so dataframe is set up the same. Overwrite below with calculation

for column_for in df_trace_forward.columns[:]:
    # Divide each tracer concentration by the input precipitation
    df_forward[column_for] = df_trace_forward[column_for] / transposed_df_inputP[column_for].values

# Compute the sum of each column
sum_forward = df_forward.sum(axis=0)

# Append the sums as a new row to the DataFrame
df_forward_sum = df_forward.append(sum_forward, ignore_index=True)

# Apply the roll_up_on_zero function to each column
# forward_ttd = df_forward.iloc[:, :].apply(roll_up_on_zero, axis=0)
forward_ttd = df_forward_sum.copy()
forward_ttd.reset_index(inplace=True)
forward_ttd.rename(columns={'index': 'Months'}, inplace=True)
# forward_ttd.to_csv(save_path_forward + f'{name_id}_ForwardTTD.csv', index=False)

#%%
## Make the Backward TTD
df_output = df_total['TotalStreamflow']
transposed_df_output = df_output.to_frame().T #transpose to make division over each column
transposed_df_output.columns = df_trace.columns[:] # Make column names the same for loop below
df_drop = df_trace.iloc[:, :]
tracer_transpose = df_drop.T 

flipped_tracer_transpose = tracer_transpose[::-1]
# df_trace_backward = flipped_tracer_transpose.iloc[:, :].apply(roll_up_on_zero, axis=0)

# Apply the function to each column
df_trace_backward = flipped_tracer_transpose.apply(roll_up_on_zero, axis=0)

df_trace_backward.columns = df_trace.columns[:] # Make column names the same for loop below
# df_trace_backward.to_csv(save_path + "Backward_TracedStreamflow.csv", index=False)

transposed_df_output.columns = df_trace.columns[:] # Make column names the same for loop below
# transposed_df_output.to_csv(save_path + "Backward_TotalStreamflow.csv", index=False)

## Divide each column by single value of precipitation
df_backward = df_trace_backward.copy()
for column_back in df_trace_backward.columns[:]:
    df_backward[column_back] = df_trace_backward[column_back] / transposed_df_output[column_back].values

# Compute the sum of each column
sum_backward = df_backward.sum(axis=0)
# Append the sums as a new row to the DataFrame
df_backward_sum = df_backward.append(sum_backward, ignore_index=True)

# Assign the column names of forward_ttd to df_backward
df_backward_sum.reset_index(inplace=True)
backward_ttd = df_backward_sum.drop(df_backward_sum.columns[0], axis=1)
backward_ttd.reset_index(inplace=True)
backward_ttd.rename(columns={'index': 'Months'}, inplace=True)
# backward_ttd.to_csv(save_path_backward + f'{name_id}_BackwardTTD.csv', index=False)


#%%
## Normalize and get the PDF and CDFs
forward_ttd  = pd.read_csv(save_path_forward + f'{name_id}_ForwardTTD.csv')
backward_ttd = pd.read_csv(save_path_backward + f'{name_id}_BackwardTTD.csv')

forward_ttd.replace(0, np.nan, inplace=True) # add Nans because dont want to consider 0's
backward_ttd.replace(0, np.nan, inplace=True)

## Thresholds based on years or months of data to use
forward_thres  = forward_ttd.iloc[:, :] # Could alter to define years of data to use
backward_thres = backward_ttd.iloc[:, :] 

## Axis = 0  is the ROW and Axis = 1 is the Column!
forward_thres  = forward_thres.iloc[:-1]  # Drop the last value of the SUM
backward_thres = backward_thres.iloc[:-1]  
forward_thres = forward_thres.iloc[:,1:] # Drop first column
backward_thres = backward_thres.iloc[:,1:]

## Find the median of 3 years worth of TTDs
MTT_forward  = forward_thres.iloc[:,:].median(axis=1) # find median of each row excluding months in 1st column
MTT_forward  = MTT_forward.iloc[:-1]  # Drop the last value of the SUM
MTT_backward = backward_thres.iloc[:,:].median(axis=1)
MTT_backward = MTT_backward.iloc[:-1]

## Get the CDF of the MTT's
CDF_forward  = MTT_forward.sum()
CDF_backward = MTT_backward.sum()

## Normalize values for finaly TTD or PDF. Should sum to 1!
forward_norm = MTT_forward.div(CDF_forward)
F_CDF_norm = forward_norm.cumsum() #Make sure sum to 1
forward_norm_df = pd.DataFrame(forward_norm)

backward_norm = MTT_backward.div(CDF_backward)
B_CDF_norm = backward_norm.cumsum()

#%%
SAVE = '/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/Paper Draft/Figures/'
# Create figure and axes for the subplot layout
fig = plt.figure(figsize=(10, 10), dpi=500)

# Create subplot layout using subplot2grid
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((3, 2), (1, 0))
ax3 = plt.subplot2grid((3, 2), (1, 1))

tracer_all = df_trace.iloc[:,1:]

# Plot traced streamflow and total streamflow on ax1
ax1.plot(df_total['Date'], df_trace['Tracer_2016_01'], color='tab:blue', label='Traced 01/2016', zorder=2) # linewidth=2,
ax1.plot(df_total['Date'][24:], df_trace['Tracer_2018_01'][24:], color='tab:red', label='Traced 01/2018',  zorder=3) # Skips first 0 point
ax1.plot(df_total['Date'][48:], df_trace['Tracer_2020_01'][48:], color='tab:orange', label='Traced 01/2020', zorder=4)
# ax1.plot(df_trace['Date'], tracer_all, zorder=2)
ax1.plot(df_total['Date'], df_total['TotalStreamflow'], color='black', label='Total Streamflow', zorder=1)
ax1.set_ylabel('Streamflow (m³/month)', fontsize=10)
ax1.set_yscale('log')
ax1.grid()
# Set x-axis limits to remove whitespace
# ax1.set_xlim(df_trace['Date'].iloc[0], df_trace['Date'].iloc[-1])
start_date = df_total['Date'].iloc[0]
end_date = df_total['Date'].iloc[-1]
date_range = end_date - start_date
extra_space = date_range * 0.014  # Adjust this factor as per your preference
ax1.set_xlim(start_date - extra_space, end_date + extra_space)
ax1.set_ylim([1e-10, 1e8])
ax1.text(0.01, 1.06, 'a)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')

# Create a second y-axis for precipitation on ax1
ax2_precip = ax1.twinx()
# ax2_precip.bar(df_trace['Date'][0], df_precip['MonthlyPrecip m3'][0], color='green', width=25, label='Precipitation', alpha=0.6, zorder=0)
ax2_precip.bar(df_total['Date'], df_precip['MonthlyPrecip_depth'], color='green', width=25, label='Precipitation', alpha=0.6, zorder=0)
ax2_precip.set_ylabel('Precipitation (m³/month)', color='green', fontsize=10)
ax2_precip.set_yscale('log')
ax2_precip.set_ylim([1e6, 1e10])

# Plot the subplot on the left (sum_backward and sum_forward)
ax2.plot(sum_backward.values * 100, label=r'Backward = $\frac{Traced\ Streamflow}{Total\ Streamflow}$', color='navy')
ax2.plot(sum_forward.values * 100, label=r'Forward = $\frac{Traced\ Streamflow}{Precipitation}$', color='crimson')
ax2.set_xlabel('Months', fontsize=10)
ax2.set_ylabel('Percent of Summed Mass', fontsize=10)
ax2.set_xlim(0, 83)  # Adjust xlim
ax2.set_xticks(np.arange(0, 96, 12))  # Set xticks every 12 ticks
ax2.grid(True)
# ax2.axvline(x=20, color='black', linewidth=1, ymax=0.82)  # Add a horizontal dotted line on the x-axis
ax2.text(0.04, 1.06, 'b)', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top')

# Plot the subplot on the right (forward_norm, backward_norm, F_CDF_norm, B_CDF_norm)
ax3.plot(backward_norm * 100, label='Backward CDF', color='navy')
# ax3.plot(forward_norm * 100, label='Forward CDF', color='crimson')
ax3.set_xlabel('Months',fontsize=10)
ax3.set_ylabel('Probability of Mass Traced', fontsize=10)
ax3.set_xlim(0, 83)
ax3.grid(True)
ax3.set_xticks(np.arange(0, 96, 12))  # Set xticks every 12 ticks

# Create a secondary axis for the CDF
ax3_cdf = ax3.twinx()
ax3_cdf.plot(B_CDF_norm * 100, label='Backward CDF', linestyle='--', color='navy')
# ax3_cdf.plot(F_CDF_norm * 100, label='Forward CDF', linestyle='--', color='crimson')
ax3_cdf.set_ylabel('Percent of Total Mass Traced', fontsize=10)
ax3.text(0.04, 1.06, 'c)', transform=ax3.transAxes, fontsize=12, fontweight='bold', va='top')

# Combine legends for all axes
# t_lines, t_labels       = ax1.get_legend_handles_labels()
# p_lines, p_labels       = ax2_precip.get_legend_handles_labels()
frac_lines, frac_labels = ax2.get_legend_handles_labels()
pdf_lines, pdf_labels   = ax3.get_legend_handles_labels()
cdf_lines, cdf_labels   = ax3_cdf.get_legend_handles_labels()

# Combine all handles and labels
# top_lines = t_lines + p_lines
# top_labels = t_labels + p_labels
bottom_lines = pdf_lines + cdf_lines
bottom_labels = frac_labels + pdf_labels + cdf_labels

# Create the combined legend
# ax1.legend(t_lines, t_labels, bbox_to_anchor=(0.45, 0.62), borderaxespad=0, fontsize=10, ncol=2)  # (0.41, 0.67)
# ax2.legend(frac_lines, frac_labels, bbox_to_anchor=(0.2, 0.3), borderaxespad=0, fontsize=10)
ax3.legend(bottom_lines, bottom_labels,bbox_to_anchor=(0.4, 0.95), borderaxespad=0, fontsize=10)

ax1.set_title(f'{name_id}: WT-WRF-Hydro Tracer Tracking', fontsize=12)

plt.tight_layout()
# plt.savefig(SAVE + f'Figure3_TTDs_{name_id}.png', dpi=500)

#%%

# Create figure 
fig = plt.figure(figsize=(10, 10), dpi=500)

sum_MTT = np.sum(MTT_backward)
plt.plot(MTT_backward*100)
plt.annotate(f'Sum: {sum_MTT}', 
              xy=(0.92, 0.86), 
              xycoords='axes fraction', 
              ha='right', 
              va='top')

plt.grid()
# Add title and labels
plt.title('Median Backward TTD')
plt.xlabel('Months')
plt.ylabel('Percent of Mass Traced')

#%%
## Plot each column of the tracer monthly events by the total streamflow
forward_drop = forward_ttd[:-1].copy()
columns_to_plot = forward_drop.columns[1:]

ymin = forward_drop[columns_to_plot].min().min()
ymax = forward_drop[columns_to_plot].max().max()

# Plot each column separately
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    plt.plot(forward_drop['Months'], forward_drop[column], label=column)
    plt.xlabel('Months')
    plt.ylabel('Percent of Mass Traced')
    plt.ylim(ymin, ymax)
    plt.title(f'{name_id}: Forward TTD {column}')
    plt.legend()
    plt.grid(True)
    
    # Annotate with the sum of the last column
    last_column_sum = round(forward_ttd[column].iloc[-1],2)  # Sum of the last column
    plt.annotate(f'Sum: {last_column_sum}', 
                 xy=(0.92, 0.86), 
                 xycoords='axes fraction', 
                 ha='right', 
                 va='top')
    
    ## Save each plot
    # filename = save_path_forward + f'{column}.png'
    # plt.savefig(filename, dpi=300, bbox_inches='tight') 
    
#%%
## Plot each column of the tracer monthly events by the total streamflow
backward_drop = backward_ttd[:-1].copy()
columns_to_plot = backward_drop.columns[1:]

columns_to_plot = backward_ttd.columns[1:]
save_path_back = f'/Users/butl842/Library/CloudStorage/OneDrive-PNNL/Desktop/PNNL-OSU-Outside/Plots/Tracer/{name_id}/TTD/Backward/'
ymin = backward_drop[columns_to_plot].min().min()
ymax = backward_drop[columns_to_plot].max().max()

# Plot each column separately
for column in columns_to_plot:
    plt.figure(figsize=(10, 6))
    plt.plot(backward_drop['Months'], backward_drop[column], label=column)
    plt.xlabel('Months')
    plt.ylabel('Percent of Mass Traced')
    plt.ylim(ymin, ymax+0.1)
    plt.title(f'{name_id}: Backward TTD {column}')
    plt.legend()
    plt.grid(True)
    
    # Annotate with the sum of the last column
    last_column_sum = round(backward_ttd[column].iloc[-1],2)  # Sum of the last column
    plt.annotate(f'Sum: {last_column_sum}', 
                 xy=(0.92, 0.86), 
                 xycoords='axes fraction', 
                 ha='right', 
                 va='top')
    
    
    ## Save each plot
    # filename = save_path_back + f'{column}.png'
    # plt.savefig(filename, dpi=300, bbox_inches='tight') 
    
#%%
# Plot both sum_backward and sum_forward on the same plot
plt.figure(figsize=(10, 6))
plt.plot(sum_backward.values*100, label='Sum Backward', color='crimson')
# plt.plot(sum_forward.values*100, label='Sum Forward', color='orange')
plt.xlabel('Months')
plt.ylabel('Percent of Summed Mass')
plt.title(f'{name_id}', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the plot
# plt.savefig(save_path + 'Sum_Forward_Backward.png', dpi=300)

#%%
## Make all 72 plots into a GIF for FORWARD!!

import imageio
import os

# Create a list to store the filenames of the saved plots
plot_filenames = []

# Directory where the plots are saved
save_directory = 'plots'

# Iterate over the saved plots and append their filenames to the list
for filename in os.listdir(save_path_forward):
    if filename.endswith('.png'):
        plot_filenames.append(os.path.join(save_path_forward, filename))

# Sort the filenames to ensure proper ordering
plot_filenames.sort()

# Create a list to store images
images = []

# Read each plot file and append it to the list of images
for filename in plot_filenames:
    images.append(imageio.imread(filename))

# Output GIF filename
# gif_filename = os.path.join(save_path, 'HOPB_ForwardTTD.gif')

# Save the list of images as a GIF
# imageio.mimsave(gif_filename, images, duration=0.5)  # Duration is in seconds between frames


#%%
## Make all 72 plots into a GIF for BACKWARD!!
import imageio
import os

# Create a list to store the filenames of the saved plots
plot_filenames = []

# Directory where the plots are saved
save_directory = 'plots'

# Iterate over the saved plots and append their filenames to the list
for filename in os.listdir(save_path_back):
    if filename.endswith('.png'):
        plot_filenames.append(os.path.join(save_path_back, filename))

# Sort the filenames to ensure proper ordering
plot_filenames.sort()

# Create a list to store images
images = []

# Read each plot file and append it to the list of images
for filename in plot_filenames:
    images.append(imageio.imread(filename))

# Output GIF filename
# gif_filename = os.path.join(save_path, 'HOPB_BackwardTTD.gif')

# Save the list of images as a GIF
# imageio.mimsave(gif_filename, images, duration=0.5)  # Duration is in seconds between frames

