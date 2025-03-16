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

ui = pd.read_csv('ui.csv', index_col=0)

patexp = ui.groupby('inventor_id')['patent_id'].nunique()

classes = pd.get_dummies(ui, columns=['class']).groupby('inventor_id').sum()

patexp.to_csv('patexp.csv')
classes.to_csv('classes.csv')