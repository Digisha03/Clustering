#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 15:52:37 2019

@author: digisha
"""

import os
import numpy as np
import pandas as pd

from pandas import ExcelWriter
from pandas import ExcelFile

from datetime import datetime

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from matplotlib.dates import DateFormatter
from matplotlib.dates import HourLocator
import matplotlib.dates as mdates

from scipy.interpolate import interp1d
import scipy.interpolate as spi
from scipy.interpolate import splrep, splev

import statistics as s

file_dir = os.path.join('/Users/digisha/Downloads/Fatigue_project/data')
file_name = os.path.join(file_dir, 'Sample_Data.xlsx')
df = pd.read_excel(file_name, sheetname='Sheet1')
df.head()

#changing datatypes, time stamp in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

#only depression fatigue questionnare considererd
mask = (df['question_id'] >= 1600) & (df['question_id'] < 1700)
df_16 = df.loc[mask].reset_index(drop=True)
df_16['time'] = [t.time() for t in df_16['timestamp']]
#df_16['time'] = df_16['time'].apply(lambda x: x.strftime('%H:%M:%S'))
df_16['answer']=df_16.answer.astype('int64')
df_16.head()

df_16['time'] = df_16['time'].apply(lambda x: x.strftime('%H:%M:%S'))

"""mask = (df_16['subject_id'] == 6)
df_16_6 = df_16.loc[mask].reset_index(drop=True)

mask = (df_16['subject_id'] == 27)
df_16_27 = df_16.loc[mask].reset_index(drop=True)"""

#df_16[''] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
dayOfWeek={0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
df_16['weekday'] = df_16['timestamp'].dt.dayofweek.map(dayOfWeek)
df_16.head()

df_16['weekend'] = 0
for i in range(len(df_16)):
    if df_16['weekday'][i] == 'Saturday' or df_16['weekday'][i] == 'Sunday':
        df_16['weekend'][i] = 1
    else:
        df_16['weekend'][i] = 0
df_16.head(5)

datetime_object = datetime.strptime('00:00:00', '%H:%M:%S')
df_16['answer']=df_16.answer.astype('int64')
df_16['time'] = 0.00
for i in range(len(df_16)):
    df_16['time'][i] = (df_16['timestamp'][i] - datetime_object).seconds/(60*60)
    df_16['time'][i] = df_16['time'][i].round(2)
    
df_16.head()

mask = (df_16['weekend'] == 0)
df_day = df_16.loc[mask].reset_index(drop=True)
mask = (df_16['weekend'] == 1)
df_end = df_16.loc[mask].reset_index(drop=True)
df_end.head()

mask = (df_day['question_id'] == 1600)
df_day_1600 = df_day.loc[mask].reset_index(drop=True)

mask = (df_day['question_id'] == 1601)
df_day_1601 = df_day.loc[mask].reset_index(drop=True)
mask = (df_day['question_id'] == 1602)
df_day_1602 = df_day.loc[mask].reset_index(drop=True)

mask = (df_day['question_id'] == 1603)
df_day_1603 = df_day.loc[mask].reset_index(drop=True)

mask = (df_end['question_id'] == 1600)
df_end_1600 = df_end.loc[mask].reset_index(drop=True)

mask = (df_end['question_id'] == 1601)
df_end_1601 = df_end.loc[mask].reset_index(drop=True)

mask = (df_end['question_id'] == 1602)
df_end_1602 = df_end.loc[mask].reset_index(drop=True)

mask = (df_end['question_id'] == 1603)
df_end_1603 = df_end.loc[mask].reset_index(drop=True)

df_end_1603.head(10)

#plt.subplot(2, 2, 1)
plt.ylim(0, 10)
plt.yticks(np.arange(0, 10, 1))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('end - 1603')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
#sns.lineplot(x = 'time', y="answer", color = 'crimson', data=df_end_1603,  marker = 'd')

sns.lineplot(x = 'time', y="answer", hue = "subject_id", data = df_end_1603, 
             palette=sns.color_palette('Dark2', n_colors=2), 
             marker = 'o')
#plt.legend(loc='upper right')
plt.xlabel('Time (days)')
plt.tight_layout()
plt.show()

#plt.subplot(2, 2, 1)
plt.ylim(0, 10)
plt.yticks(np.arange(0, 10, 1))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('day - 1600')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
#sns.lineplot(x = 'time', y="answer", color = 'crimson', data=df_end_1603,  marker = 'd')

sns.lineplot(x = 'time', y="answer", hue = "subject_id", data = df_day_1600, 
             palette=sns.color_palette('Dark2', n_colors=2), 
             marker = 'o')
#plt.legend(loc='upper right')
plt.xlabel('Time (days)')
plt.tight_layout()
plt.show()

plt.figure(figsize = (30,15))
plt.suptitle('Subject 6 (time v/s answer(absolute) in 24 hours) for weekday', fontsize=15)

plt.subplot(2, 2, 1)
ndf = df_day_1600.loc[df_day_1600['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Fatigue - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 2)
ndf = df_day_1601.loc[df_day_1601['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Depression - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 3)
ndf = df_day_1602.loc[df_day_1602['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Anxiety - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 4)
ndf = df_day_1603.loc[df_day_1603['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('pain - 1603')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.show()
#plt.legend(loc='upper right')

plt.figure(figsize = (30,15))
plt.suptitle('Subject 27 (time v/s answer(absolute) in 24 hours for weekday)', fontsize=15)

plt.subplot(2, 2, 1)
ndf = df_day_1600.loc[df_day_1600['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Fatigue - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 2)
ndf = df_day_1601.loc[df_day_1601['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Depression - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 3)
ndf = df_day_1602.loc[df_day_1602['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Anxiety - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 4)
ndf = df_day_1603.loc[df_day_1603['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('pain - 1603')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
#plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.show()
#plt.legend(loc='upper right')

plt.figure(figsize = (30,15))
plt.suptitle('Subject 6 (time v/s answer(absolute) in 24 hours for weekend)', fontsize=15)

plt.subplot(2, 2, 1)
ndf = df_end_1600.loc[df_end_1600['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Fatigue - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 2)
ndf = df_end_1601.loc[df_end_1601['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Depression - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 3)
ndf = df_end_1602.loc[df_end_1602['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Anxiety - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 4)
ndf = df_end_1603.loc[df_end_1603['subject_id'] == 6].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('pain - 1603')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.show()
#plt.legend(loc='upper right')

plt.figure(figsize = (30,15))
plt.suptitle('Subject 27 (time v/s answer(absolute) in 24 hours for weekend)', fontsize=15)

plt.subplot(2, 2, 1)
ndf = df_end_1600.loc[df_end_1600['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Fatigue - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 2)
ndf = df_end_1601.loc[df_end_1601['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Depression - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 3)
ndf = df_end_1602.loc[df_end_1602['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('Anxiety - 1600')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')

plt.subplot(2, 2, 4)
ndf = df_end_1603.loc[df_end_1603['subject_id'] == 27].reset_index(drop=True)
ndf = ndf.sort_values(by=['time']).reset_index(drop=True)
x = ndf['time'].values
y = ndf['answer'].values
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
#plt.xticks(np.arange(0, max(x), 1),rotation='45')
plt.title('pain - 1603')
sns.lineplot(x = x, y=y, color = 'skyblue', data= ndf,  marker = 'd')
k = ndf.answer.ewm(span=2, adjust=False).mean()
#plt.plot(x,k,'green')
xnew = np.linspace(0,24,60)
intfunc = spi.interp1d(x,y,fill_value="extrapolate")
y_interp = intfunc(xnew)
#plt.plot(xnew,y_interp,'red', label='interp/extrap')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.show()
#plt.legend(loc='upper right')

plt.figure(figsize = (30,15))
plt.suptitle('Subject 27 (time v/s answer(mean and standard deviation) in 24 hours for weekday)', fontsize=15)

plt.subplot(2, 2, 1)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1600.loc[df_day_1600['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Fatigue - 1600')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 2)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1601.loc[df_day_1601['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Depression - 1601')
plt.errorbar(time, mean, std, linestyle='-', marker='o')


plt.subplot(2, 2, 3)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1602.loc[df_day_1602['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Anxiety - 1602')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 4)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1603.loc[df_day_1603['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Pain - 1603')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.show()

plt.figure(figsize = (30,15))
plt.suptitle('Subject 6 (time v/s answer(mean and standard deviation) in 24 hours for weekday)', fontsize=15)

plt.subplot(2, 2, 1)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1600.loc[df_day_1600['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Fatigue - 1600')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 2)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1601.loc[df_day_1601['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Depression - 1601')
plt.errorbar(time, mean, std, linestyle='-', marker='o')


plt.subplot(2, 2, 3)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1602.loc[df_day_1602['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Anxiety - 1602')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 4)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_day_1603.loc[df_day_1603['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Pain - 1603')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.show()

plt.figure(figsize = (30,15))
plt.suptitle('Subject 27 (time v/s answer(mean and standard deviation) in 24 hours for weekend)', fontsize=15)

plt.subplot(2, 2, 1)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1600.loc[df_end_1600['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Fatigue - 1600')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 2)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1601.loc[df_end_1601['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Depression - 1601')
plt.errorbar(time, mean, std, linestyle='-', marker='o')


plt.subplot(2, 2, 3)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1602.loc[df_end_1602['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Anxiety - 1602')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 4)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1603.loc[df_end_1603['subject_id'] == 27].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Pain - 1603')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.show()

plt.figure(figsize = (30,15))
plt.suptitle('Subject 6 (time v/s answer(mean and standard deviation) in 24 hours for weekend)', fontsize=15)

plt.subplot(2, 2, 1)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1600.loc[df_end_1600['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Fatigue - 1600')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 2)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1601.loc[df_end_1601['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Depression - 1601')
plt.errorbar(time, mean, std, linestyle='-', marker='o')


plt.subplot(2, 2, 3)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1602.loc[df_end_1602['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Anxiety - 1602')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.subplot(2, 2, 4)
k = 0.00
list1 = []
list2 = []
list3 = []
list4 = []

ndf = df_end_1603.loc[df_end_1603['subject_id'] == 6].reset_index(drop=True)
for i in range(len(ndf)):
    k = ndf['answer'][i]
    if ndf['time'][i] < 6.00:
        list1.append(k)
    elif ndf['time'][i] >= 6.00 and ndf['time'][i] < 12.00:
        list2.append(k)
    elif ndf['time'][i] >= 12.00 and ndf['time'][i] < 18.00:
        list3.append(k)
    elif ndf['time'][i] >= 18.00 and ndf['time'][i] < 24.00:
        list4.append(k)
    else:
        continue
#import mean 
mean = [sum(list1)/len(list1),sum(list2)/len(list2),sum(list3)/len(list3),sum(list4)/len(list4)]
mean = [ round(elem, 2) for elem in mean ]
std = [np.std(list1).round(2),np.std(list2).round(2),np.std(list3).round(2),np.std(list4).round(2)]

time = [3,9,15,21]
plt.ylim(0, 10)
plt.xlim(0, 24)
plt.xlabel('Time (hours)')
plt.ylabel('Answer')
plt.yticks(np.arange(0, 10, 1))
plt.xticks(np.arange(0, 27, 3))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
plt.title('Pain - 1603')
plt.errorbar(time, mean, std, linestyle='-', marker='o')

plt.show()