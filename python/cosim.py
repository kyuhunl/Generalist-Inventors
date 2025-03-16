import os
import zipfile as zip
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import ast
import datetime
import dateparser as dp
from sklearn.metrics.pairwise import cosine_similarity
from concentrationMetrics import Index

os.chdir('E:\data')

gcpd = pd.read_csv('GCPD_granular_data.txt')
pi = pd.read_csv('patent_inventor.tsv', delimiter='\t')
ipc = pd.read_csv('ipc3.csv', usecols=['patent_id','class']).drop_duplicates()
top10 = ['Novartis AG', 'Pfizer Inc', 'Roche Holding AG', 'Sanofi', 'Merck & Co Inc.', 'Johnson & Johnson', 'Glaxosmithkline PLC', 'Astrazeneca PLC' ,'Gilead Sciences Inc', 'AbbVie Inc']

# patents from top 10 companies
top10pat = gcpd[gcpd['conml'].isin(top10)]
# convert date to datetime
top10pat['date_issue'] = top10pat['date_issue'].apply(dp.parse)
top10pat['date_filing'] = top10pat['date_filing'].apply(dp.parse)

#save
top10pat.to_csv('top10pat.csv')

# selecting sampleframe (issue date 2010-2015)
sampleframe = top10pat[(top10pat['date_issue']>=datetime.datetime(2010,1,1)) & (top10pat['date_issue']<datetime.datetime(2016,1,1))]
sampleframe = sampleframe.rename(columns={'nr_pt':'patent_id'})
sampleframe['patent_id'] = sampleframe['patent_id'].astype(str)
sampleframe = sampleframe.merge(pi, how='left', on='patent_id')

#save
sampleframe.to_csv('sampleframe.csv')

# extracting unique inventors
ui = pd.DataFrame(sampleframe['inventor_id'].unique(), columns=['inventor_id'])
# merge with pi to get portfolio for each inventor
ui = ui.merge(pi[['patent_id','inventor_id']], how='left', on='inventor_id')
# merge with ipc to get class
ui = ui.merge(ipc, how='left', on='patent_id')
ui = ui.drop_duplicates()

#save
ui.to_csv('ui.csv')

# class information for each patent of each inventor + dummies + #class
pf = pd.concat([ui, pd.get_dummies(ui['class'])], axis=1).merge(ui.dropna(subset=['class']).drop_duplicates(subset=['patent_id','class']).groupby('patent_id')['class'].count(), how='left', on='patent_id')

# dividing dummis by #class to gain weighted dummy
for i in pf.columns:
    if i in ['inventor_id', 'patent_id', 'class_x', 'class_y']:
        continue
    pf[i] = pf[i]/pf['class_y']

#save
pf.to_csv('pf.csv')

# weighted class count of each inventors
vectors = pf[pf.columns.difference(['patent_id', 'class_x', 'class_y'])].groupby('inventor_id').sum()

#save
vectors.to_csv('vectors.csv')

# merge with sampleframe to get portfolio of each inventor in each patent
p_v = sampleframe[{'patent_id','inventor_id'}].merge(vectors,how='left',on='inventor_id').drop(columns=['inventor_id']).set_index('patent_id')

#save
p_v.to_csv('p_v.csv')

# groupby patent
grouped = p_v.groupby('patent_id')

# cosine similarity btw inventors for each patent
cosim = grouped.apply(cosine_similarity)

#save
cosim.to_csv('cosim.csv')

# getting the median and mean similarity
ivs = pd.DataFrame({'median': cosim.apply(lambda x: np.median(x[np.triu_indices(len(x), k=1)])), 'mean': cosim.apply(lambda x: np.mean(x[np.triu_indices(len(x), k=1)]))})

#save
ivs.to_csv('ivs.csv')