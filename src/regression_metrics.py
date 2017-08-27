import scipy.stats as stats
import pandas
import numpy as np

FILENAME='../data/asset-v1-PennX_BDE1x_2017_regressor-data-asgn2.csv'

df = pandas.read_csv(FILENAME)

x = df['data']
y = df['predicted (model)']

print 'PEARSON R', stats.pearsonr(x, y)


print 'RMSE: ', np.sqrt(((x - y) ** 2).mean())

print 'MAE: ', abs((x - y)).mean()
