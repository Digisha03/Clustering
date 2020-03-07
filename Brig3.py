#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:31:31 2019

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

file_dir = os.path.join('/Users/digisha/Downloads/Fatigue_project/data')
file_name = os.path.join(file_dir, 'Sample_Data.xlsx')
df = pd.read_excel(file_name, sheetname='Sheet1')
df.head()

#changing datatypes, time stamp in datetime format
df['timestamp'] = pd.to_datetime(df['timestamp'])

#only depression fatigue questionnare considererd
mask = (df['question_id'] >= 1600) & (df['question_id'] < 1700)
df_16 = df.loc[mask].reset_index(drop=True)
datetime_object = datetime.strptime('00:00:00', '%H:%M:%S')
df_16['answer']=df_16.answer.astype('int64')
df_16.head()

df_16['time'] = 0.00
for i in range(len(df_16)):
    df_16['time'][i] = (df_16['timestamp'][i] - datetime_object).seconds/(60*60)
    df_16['time'][i] = df_16['time'][i].round(2)
    
df_16.head()

mask = (df_16['subject_id'] == 6)
df_16_6 = df_16.loc[mask].reset_index(drop=True)

mask = (df_16['subject_id'] == 27)
df_16_27 = df_16.loc[mask].reset_index(drop=True)

df_16_61 = df_16_6[(df_16_6['time'] >= 0) & (df_16_6['time'] < 6)]
df_16_62 = df_16_6[(df_16_6['time'] >= 6) & (df_16_6['time'] < 12)]
df_16_63 = df_16_6[(df_16_6['time'] >= 12) & (df_16_6['time'] < 18)]
df_16_64 = df_16_6[(df_16_6['time'] >= 18) & (df_16_6['time'] < 24)]

df_16_271 = df_16_27[(df_16_27['time'] >= 0) & (df_16_27['time'] < 6)]
df_16_272 = df_16_27[(df_16_27['time'] >= 6) & (df_16_27['time'] < 12)]
df_16_273 = df_16_27[(df_16_27['time'] >= 12) & (df_16_27['time'] < 18)]
df_16_274 = df_16_27[(df_16_27['time'] >= 18) & (df_16_27['time'] < 24)]

plt.figure(figsize = (15,10))
plt.subplot(2, 2, 1)
plt.ylim(0, 9)
plt.xlim(0, 6)
plt.title('Early Morning-Subject 6-1')
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(0, 7, 1))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_61, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.subplot(2, 2, 2)
plt.ylim(0, 9)
plt.xlim(6, 12)
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(5, 13, 1))
#plt.xticks(np.arange(18, 25, 1))
plt.title('Morning-Subject 6-2')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_62, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.subplot(2, 2, 3)
plt.ylim(0, 9)
plt.xlim(12, 18)
#plt.xticks(np.arange(0, max(x), 0.5),rotation='vertical')
plt.title('Afternoon -Subject 6-3')
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(12, 19, 1))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_63, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.subplot(2, 2, 4)
plt.ylim(0, 9)
plt.xlim(18, 24)
plt.title('Evening-Subject 6-4')
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(18, 25, 1))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_64, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize = (15,10))
plt.subplot(2, 2, 1)
plt.ylim(0, 9)
plt.xlim(0, 6)
plt.title('Early Morning-Subject 27-1')
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(0, 7, 1))
plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_271, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.subplot(2, 2, 2)
plt.ylim(0, 9)
plt.xlim(6, 12)
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(5, 13, 1))
#plt.xticks(np.arange(18, 25, 1))
plt.title('Morning-Subject 27-2')
plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_272, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.subplot(2, 2, 3)
plt.ylim(0, 9)
plt.xlim(12, 18)
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(12, 19, 1))
plt.title('Afternoon -Subject 27-3')

plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_273, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.subplot(2, 2, 4)
plt.ylim(0, 9)
plt.xlim(18, 24)
plt.title('Evening-Subject 27-4')
plt.yticks(np.arange(0, 9, 1))
plt.xticks(np.arange(18, 25, 1))

plt.grid(color='grey', linestyle='-', linewidth=0.25)
sns.scatterplot(x = 'time', y="answer", hue = "question_id", data=df_16_274, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.show()


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

mask = (df_16['subject_id'] == 6)
df_16_6 = df_16.loc[mask].reset_index(drop=True)

mask = (df_16['subject_id'] == 27)
df_16_27 = df_16.loc[mask].reset_index(drop=True)

#dividing into 4 time zones

df_16_61 = df_16_6[(df_16_6['time'] >= '00:00:00') & (df_16_6['time'] < '06:00:00')]
df_16_62 = df_16_6[(df_16_6['time'] >= '06:00:00') & (df_16_6['time'] < '12:00:00')]
df_16_63 = df_16_6[(df_16_6['time'] >= '12:00:00') & (df_16_6['time'] < '18:00:00')]
df_16_64 = df_16_6[(df_16_6['time'] >= '18:00:00') & (df_16_6['time'] < '24:00:00')]

df_16_271 = df_16_27[(df_16_27['time'] >= '00:00:00') & (df_16_27['time'] < '06:00:00')]
df_16_272 = df_16_27[(df_16_27['time'] >= '06:00:00') & (df_16_27['time'] < '12:00:00')]
df_16_273 = df_16_27[(df_16_27['time'] >= '12:00:00') & (df_16_27['time'] < '18:00:00')]
df_16_274 = df_16_27[(df_16_27['time'] >= '18:00:00') & (df_16_27['time'] < '24:00:00')]
#df_16_62.head()
#df_16_61.head()

plt.figure(figsize = (15,10))
#plt.subplot(2, 2, 1)
x = df_16_61['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 1)
ax.xaxis.set_major_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 6 - 00:00 to 06:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_61, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

#plt.subplot(2, 2, 2)
x = df_16_62['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 2)
#ax.plot(x, df_16_61['answer'])
ax.xaxis.set_major_locator(mdates.HourLocator())
#ax.xaxis.set_major_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 6 - 06:00 to 12:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_62, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

#plt.subplot(2, 2, 3)
x = df_16_63['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 3)
#ax.plot(x, df_16_61['answer'])
ax.xaxis.set_major_locator(mdates.HourLocator())
#ax.xaxis.set_major_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 6 - 12:00 to 18:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_63, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

#plt.subplot(2, 2, 4)
x = df_16_64['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 4)
#ax.plot(x, df_16_61['answer'])
ax.xaxis.set_major_locator(mdates.HourLocator())
#ax.xaxis.set_major_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 6 - 18:00 to 24:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_64, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize = (15,10))
#plt.subplot(2, 2, 1)
x = df_16_271['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 1)
ax.xaxis.set_major_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 27 - 00:00 to 06:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_271, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

#plt.subplot(2, 2, 2)
x = df_16_272['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 2)
#ax.plot(x, df_16_61['answer'])
ax.xaxis.set_major_locator(mdates.HourLocator())
#ax.xaxis.set_major_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 27 - 06:00 to 12:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_272, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

#plt.subplot(2, 2, 3)
x = df_16_273['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 3)
#ax.plot(x, df_16_61['answer'])
ax.xaxis.set_major_locator(mdates.HourLocator())
#ax.xaxis.set_major_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 27 - 12:00 to 18:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_273, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

#plt.subplot(2, 2, 4)
x = df_16_274['timestamp'].apply(lambda x: x.replace(year=1967, month=6, day=25)).tolist()
ax = plt.subplot(2, 2, 4)
#ax.plot(x, df_16_61['answer'])
ax.xaxis.set_major_locator(mdates.HourLocator())
#ax.xaxis.set_major_locator(HourLocator())
ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
plt.title('Subject 27 - 18:00 to 24:00')
sns.lineplot(x = x, y="answer", hue = "question_id", data=df_16_274, 
             palette=sns.color_palette('Dark2', n_colors=4), 
             marker = 'o')
plt.legend(loc='upper right')

plt.show()