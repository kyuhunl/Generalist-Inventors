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

# chunking up the file
read = pd.read_csv('uspatentcitation.tsv', delimiter='\t', usecols=['patent_id','citation_id'], chunksize=1000000)

# assuming we have sampleframe
sampleframe = pd.read_csv('sampleframe.csv', index_col=0)
patlist = list(sampleframe['patent_id'].unique())

citing = pd.concat(chunk[chunk['patent_id'].isin(patlist)] for chunk in read)
cited = pd.concat(chunk[chunk['citation_id'].isin(patlist)] for chunk in read)

citing.to_csv('citing.csv')
cited.to_csv('cited.csv')