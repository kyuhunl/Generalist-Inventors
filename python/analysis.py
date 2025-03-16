import os
import zipfile as zip
import pandas as pd
import csv
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

os.chdir('E:\data')

# open files
# patent-inventor data
sampleframe = pd.read_csv('sampleframe.csv', index_col=0, usecols=['patent_id', 'conml', 'date_issue', 'date_filing', 'inventor_id']).reset_index()
# inventor data
hhis = pd.read_csv('hhis.csv', index_col=0)
# patent data
p_v = pd.read_csv('p_v.csv', index_col=0)
# cosim = pd.read_csv('cosim.csv', index_col=0)
# cosim = cosim.apply(ast.literal_eval)
ivs = pd.read_csv('ivs.csv', index_col=0)
bwdcit = pd.read_csv('bwdcit.csv', index_col=0)
fwdcit = pd.read_csv('fwdcit.csv', index_col=0)
originality = pd.read_csv('originality.csv', index_col=0)
generality = pd.read_csv('generality.csv', index_col=0)
patexp = pd.read_csv('patexp.csv', index_col=0)
classes = pd.read_csv('classes.csv', index_col=0)


# dummy for generalists
hhis['isgen'] = (hhis['hhi']<=0.2).astype(int)

# getting teamsize
teamsize = sampleframe.groupby('patent_id')['inventor_id'].nunique()

# getting team characteristic (average hhi)
teams = pd.DataFrame({'patent_id': sampleframe['patent_id'], 'inventor_id':sampleframe['inventor_id']})
teamchar = teams.merge(hhis, how='left', on='inventor_id').groupby('patent_id').mean()
teamchar['numgen'] = teams.merge(hhis, how='left', on='inventor_id').groupby('patent_id').sum().isgen
teamchar['yesgen'] = (teamchar['numgen']>0).astype(int)

#getting cosim, distance measures
grouped = p_v.groupby('patent_id')
# cosine similarity btw inventors for each patent
cosim = grouped.apply(cosine_similarity)
ivs = pd.DataFrame({'meddist': 1 - cosim.apply(lambda x: np.median(x[np.triu_indices(len(x), k=1)])), 'meandist': 1 - cosim.apply(lambda x: np.mean(x[np.triu_indices(len(x), k=1)])), 'mindist': 1 - cosim.apply(lambda x: np.amax(x[np.triu_indices(len(x), k=1)], initial=-1)), 'maxdist': 1 - cosim.apply(lambda x: np.amin(x[np.triu_indices(len(x), k=1)], initial=2))})

# firm dummies
firms = sampleframe.drop_duplicates(subset=['patent_id','conml']).set_index('patent_id')['conml']
# firmdummy = pd.get_dummies(sampleframe.drop_duplicates(subset=['patent_id','conml']).set_index('patent_id')['conml'])

# year dummies
year = pd.DataFrame({'patent_id':sampleframe.drop_duplicates(subset=['patent_id','date_issue'])['patent_id'], 'year':pd.DatetimeIndex(sampleframe.drop_duplicates(subset=['patent_id','date_issue'])['date_issue']).year}).set_index('patent_id')['year']
# yeardummy = pd.get_dummies(pd.DataFrame({'patent_id':sampleframe.drop_duplicates(subset=['patent_id','date_issue'])['patent_id'], 'year':pd.DatetimeIndex(sampleframe.drop_duplicates(subset=['patent_id','date_issue'])['date_issue']).year}).set_index('patent_id')['year'])

#patenting experience
teamexp = sampleframe.merge(patexp, on='inventor_id',how='left').rename(columns={'patent_id_x':'patent_id', 'patent_id_y':'patexp'}).groupby('patent_id')['patexp'].sum()

#team scope
teamscope = sampleframe.merge(classes, on='inventor_id', how='left').drop(columns={'conml','date_issue','date_filing','inventor_id'}).groupby('patent_id').sum()
for i in teamscope.columns:
    teamscope[i] = (teamscope[i]>0).astype(int)
teamscope['teamscope'] = teamscope.sum(axis=1)

# frames to merge: teamchar, ivs, bwdcit, fwdcit, originality, generality, firms, year
#renaming 
# frames
bwdcit = bwdcit.rename(columns={'citation_id':'bwdcit'})
fwdcit = fwdcit.rename(columns={'citation_id':'patent_id','patent_id':'fwdcit'})
generality = generality.rename(columns = {'og':'gen'})
# frames = [teamsize, teamchar, ivs, bwdcit, fwdcit, originality, generality, firms, firmdummy, year, yeardummy]
frames = [teamsize, teamchar, ivs, bwdcit, fwdcit, originality, generality, firms, year, teamexp, teamscope['teamscope']]
analysis = reduce(lambda left, right: pd.merge(left, right, on='patent_id', how='left'), frames).set_index('patent_id')
analysis = analysis.rename(columns={'inventor_id':'teamsize'})




df = analysis.dropna()
df.describe()

#columns
# 'teamsize', 'hhi', 'isgen', 'numgen', 'yesgen', 'meddist', 'meandist', 'mindist', 'maxdist', 'bwdcit', 'fwdcit', 'og', 'gen', 'conml', 'year', 'patexp', 'teamscope'
# IV : hhi, isgen, yesgen, median, mean, max, min
# Control : teamsize
# DV : bwdcit, fwdcit, og, gen

plt.scatter(df['numgen'],df['meandist'])
plt.show()

j=0
for i in df.columns:
    j+=1
    plt.subplot(4, 5, j)
    plt.hist(df[f'{i}'])
    plt.title(f'{i}')
plt.show()

df.plot('isgen', 'fwdcit', kind='scatter')
plt.show()


sm.OLS.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + meandist + isgen:meandist + C(conml) + C(year)', df).fit().summary()
sm.OLS.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + meandist + isgen:meandist + C(conml) + C(year)', df[df['teamsize']>2]).fit().summary()

sm.OLS.from_formula('(1-fwdcit) ~ teamsize + isgen + meandist + isgen:meandist', df).fit().summary()
sm.OLS.from_formula('fwdcit ~ teamsize + isgen + meandist + isgen:meandist', df).fit().summary()


statsmodels.discrete.discrete_model.NegativeBinomial.from_formula('fwdcit ~ teamsize + isgen + meandist + isgen:meandist + C(conml) + C(year)', df, loglike_method='nb2', ).fit().summary()

df.corr(method = 'pearson')

for alpha in np.linspace(0.01, 1, 10):
    model = sm.GLM.from_formula('fwdcit ~ teamsize + isgen + meandist + isgen:meandist + C(conml) + C(year)', df, family=sm.families.NegativeBinomial(alpha=alpha))
    model_fitted = model.fit(method='newton')
    print(alpha, model_fitted.bic)
    print(model_fitted.params)

sm.GLM.from_formula('fwdcit ~ teamsize + isgen + meandist + isgen:meandist + C(conml) + C(year)', df, family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
sm.GLM.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + meandist + isgen:meandist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
sm.GLM.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + meandist + np.power(meandist,2) + isgen:meandist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
# sm.GLM.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + maxdist + np.power(maxdist,2) + isgen:maxdist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
# sm.GLM.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + maxdist + isgen:maxdist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()

sm.OLS.from_formula('gen ~ teamsize + isgen + meandist + isgen:meandist + C(conml) + C(year)', df).fit().summary()
sm.OLS.from_formula('gen ~ teamsize + isgen + np.power(isgen,2) + meandist + np.power(meandist,2) + isgen:meandist + C(conml) + C(year)', df[df['teamsize']>2]).fit().summary()
sm.OLS.from_formula('gen ~ teamsize + isgen + np.power(isgen,2) + maxdist + isgen:maxdist + C(conml) + C(year)', df[df['teamsize']>2]).fit().summary()

def omx(x):
    return 1-x


sm.GLM.from_formula('fwdcit ~ teamsize + omx(hhi) + np.power(omx(hhi),2) + meandist + np.power(meandist,2) + omx(hhi):meandist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
sm.GLM.from_formula('fwdcit ~ teamsize + hhi + np.power(hhi,2) + meandist + np.power(meandist,2) + hhi:meandist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
print(Stargazer([sm.GLM.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + meandist + np.power(meandist,2) + isgen:meandist + patexp + patexp:meandist + teamscope + teamscope:meandist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit()]).render_latex())
sm.GLM.from_formula('fwdcit ~ teamsize + isgen + np.power(isgen,2) + meandist + np.power(meandist,2) + isgen:meandist + patexp + patexp:meandist + teamscope + teamscope:meandist + C(conml) + C(year)', df[df['teamsize']>2], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()

sm.GLM.from_formula('fwdcit ~ teamsize + meandist + np.power(meandist,2) + C(conml) + C(year)', df[(df['teamsize']>2) & (df['isgen']==0)], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
sm.GLM.from_formula('fwdcit ~ teamsize + meandist + np.power(meandist,2) + C(conml) + C(year)', df[(df['teamsize']>2) & (df['isgen']>0.3)], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()

sm.GLM.from_formula('fwdcit ~ teamsize + meandist + np.power(meandist,2) + patexp + patexp:meandist + teamscope + teamscope:meandist + C(conml) + C(year)', df[(df['teamsize']>2) & (df['isgen']==0)], family=sm.families.NegativeBinomial(alpha=1)).fit().summary()
sm.OLS.from_formula('gen ~ teamsize + isgen + np.power(isgen,2) + meandist + np.power(meandist,2) + isgen:meandist + patexp + patexp:meandist + teamscope + C(conml) + C(year)', df[(df['teamsize']>2)]).fit().summary()
# rough data shows results as expected.
# lacking R-squared. need more controls.
