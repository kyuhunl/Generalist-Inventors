import os
import zipfile as zip
import pandas as pd
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
import patsy
from scipy import stats
import ast
import datetime
import dateparser as dp
from sklearn.metrics.pairwise import cosine_similarity
from concentrationMetrics import Index
from functools import reduce
import statsmodels
import statsmodels.api as sm
from stargazer.stargazer import Stargazer

os.chdir('E:\patentsview 2021')

# data
# patent info: patent id, assignee, inventors, classification, application date, grant date, claims
# inventor info: portfolio, hhi, #unique classes, patenting exp(number, year)
# team info: distance btw inventors, total experience, total scope, experience heterogeneity
# citation info: forward citations, backward citations, citing/cited patents classification
# new domain: inventor scope change, inventor hhi change
# repeated collaboration: prior instances of inventor dyads

# sample
# top 25 biopharmaceutical firms?
# need before/after 5 years
# timeframe: dependent on data availability
# how to sort assignees?
# fuzzywuzzy shit?
# 2006-2010, 2011-2015, 2016-2020 ??

# open patent
patent = pd.read_csv('patent.tsv', delimiter='\t', usecols={'id','type','date'}, dtype={'id':'string','type':'category', 'date':'string'})

# utility only
utility = patent[patent['type']=='utility'][['id','date']]

utility = utility.merge(pd.read_csv('patent_assignee.tsv', delimiter='\t', dtype={'patent_id':'string'}), how='left', left_on='id', right_on='patent_id')
utility = utility.drop(columns={'patent_id'})

utility = utility.merge(pd.read_csv('assignee.tsv', delimiter='\t', usecols={'id','organization'}), how='left', left_on='assignee_id', right_on='id')
utility = utility.drop(columns={'assignee_id','id_y'}).rename(columns={'id_x':'patent_id'})


# getting classification data
# ipc = pd.read_csv('ipcr.tsv', delimiter='\t', usecols={'patent_id','section','ipc_class','subclass','symbol_position','classification_value','action_date'}, dtype={'patent_id':'string', 'section':'category','ipc_class':'string','subclass':'category','symbol_position':'category','classification_value':'category'})
# ipc[ipc['symbol_position']=='F'].sort_values(by='action_date', axis=0)

cpc = pd.read_csv('cpc_current.tsv', delimiter='\t', usecols={'patent_id','group_id','category','sequence'}, dtype={'patent_id':'string','group_id':'category','category':'category','sequence':'int'})
primary = cpc[cpc['sequence']==0]

# matching id, primary cpc, date
full = primary.merge(utility[{'patent_id','date'}], how='inner', on='patent_id')
# full['date'] = full['date'].apply(dp.parse) #too long
full['year']=full['date'].str[:4]
full[full['year'].isin(['2011','2012','2013','2014','2015'])]
#now what?
# get sample, do sample analysis, get anything I need from full patent pool

# # patents from top 10 companies
# top10pat = gcpd[gcpd['conml'].isin(top10)]
# # convert date to datetime
# top10pat['date_issue'] = top10pat['date_issue'].apply(dp.parse)
# top10pat['date_filing'] = top10pat['date_filing'].apply(dp.parse)

# #save
# top10pat.to_csv('top10pat.csv')

# # selecting sampleframe (issue date 2010-2015)
# sampleframe = top10pat[(top10pat['date_issue']>=datetime.datetime(2010,1,1)) & (top10pat['date_issue']<datetime.datetime(2016,1,1))]
# sampleframe = sampleframe.rename(columns={'nr_pt':'patent_id'})
# sampleframe['patent_id'] = sampleframe['patent_id'].astype(str)
# sampleframe = sampleframe.merge(pi, how='left', on='patent_id')

