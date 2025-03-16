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

# open files
citing = pd.read_csv('citing.csv', index_col=0)
cited = pd.read_csv('cited.csv', index_col=0)
ipc = pd.read_csv('ipc3.csv', usecols=['patent_id','class']).drop_duplicates()
# ui = pd.read_csv('ui.csv', index_col=0)
vectors = pd.read_csv('vectors.csv', index_col=0)

cited_class = cited.merge(ipc, how='left', on='patent_id')
citing_class = citing.merge(ipc, how='left', left_on='citation_id', right_on='patent_id')
citing_class = citing_class.drop(columns='patent_id_y').rename(columns={'patent_id_x':'patent_id'})

# function to calculate hhi
def hhi(vector):
    sum = np.sum(vector)
    ss = np.sum(np.square(vector))
    return ss/(sum ** 2)

# # hhi corrected for bias
# def chhi(vector):
#     n = np.sum(vector)
#     if n<=1:
#         return np.nan
#     h = hhi(vector)
#     return (h-1/n)/(1-1/n)

# # normalized hhi
# def nhhi(vector):
#     n = len(vector)
#     h = hhi(vector)
#     return (h-1/n)/(1-1/n)


# hhis = pd.DataFrame(vectors.reset_index().inventor_id.apply(lambda x: hhi(vectors.loc[x].to_numpy())), index= vectors.index )

# calculating hhis for all inventors
hhis = pd.DataFrame(vectors.index)
hhis['hhi'] = hhis.inventor_id.apply(lambda x: hhi(vectors.loc[x].to_numpy()))
hhis.to_csv('hhis.csv')

# define generalists as hhi score 0.2 or below
generalists = hhis[hhis['hhi']<=0.2]

# backward citations
bwdcit = citing.groupby('patent_id').count().reset_index()

#save
bwdcit.to_csv('bwdcit.csv')

# forward citations
fwdcit = cited.groupby('citation_id').count().reset_index()
#save
fwdcit.to_csv('fwdcit.csv')


# calculating originality (hhis for all backward citations)
citing_class = pd.concat([citing_class, pd.get_dummies(citing_class['class'])], axis=1)
citing_class = citing_class.drop(columns=['citation_id', 'class']).set_index('patent_id')
citing_class = citing_class.groupby('patent_id').sum()

originality = pd.DataFrame(citing_class.index)
originality['og'] = 1- originality.patent_id.apply(lambda x: hhi(citing_class.loc[x].to_numpy()))

# save
originality.to_csv('originality.csv')

# calculating generality
cited_class = pd.concat([cited_class, pd.get_dummies(cited_class['class'])], axis=1)
cited_class = cited_class.drop(columns=['patent_id', 'class']).set_index('citation_id')
cited_class = cited_class.groupby('citation_id').sum()

generality = pd.DataFrame(cited_class.index)
generality['gen'] = 1- generality.citation_id.apply(lambda x: hhi(cited_class.loc[x].to_numpy()))
generality = generality.rename(columns={'citation_id':'patent_id'})

#save
generality.to_csv('generality.csv')

