import numpy as np
import pandas

FILENAME='../data/asset-v1-PennX_BDE1x_2017_classifier-data-asgn2.csv'

df = pandas.read_csv(FILENAME)

total = float(len(df))
positive = (df[df['Data'] == 'Y'] )
negative = (df[df['Data'] == 'N'] )

true_positive = len(positive[positive['Predicted (Model)']>=0.5])
true_negative = len(negative[negative['Predicted (Model)']<0.5])
false_negative = len(positive) - true_positive
false_positive = len(negative) - true_negative

precision = round(true_positive/float(true_positive + false_positive), 2)
recall = true_positive/float(true_positive + false_negative)

accuracy = (true_positive + true_negative)/total

pbb_yes = ((true_positive + false_positive)/total)*((true_positive + false_negative)/total)
pbb_no = ((true_negative + false_negative)/total)*((true_negative + false_positive)/total)

pbb_chance_agreement = pbb_yes + pbb_no
print 'pbb_chance: ', pbb_chance_agreement
kappa = (accuracy - pbb_chance_agreement)/(1 - pbb_chance_agreement)

print 'Real positive: ', len(positive)
print 'Real negative: ', len(negative)
print
print 'True positive: ', true_positive 
print 'True negative: ', true_negative
print 'False positive: ', false_positive
print 'False Negative: ', false_negative
print 'Accuracy: ', accuracy
print 'Precision: ', precision
print 'Recall: ', recall
print 'Kappa: ', kappa
