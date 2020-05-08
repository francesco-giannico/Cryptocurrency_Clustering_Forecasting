from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
import pandas as pd
import numpy as np
# Define dictionary to store our rankings
ranks = {}
# Create our function which stores the feature rankings to the ranks dictionary
"""def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))"""

#X, y = make_friedman1(n_samples=5, n_features=10, random_state=0)
csv=pd.read_csv("../preparation/preprocessed_dataset/constructed/max_abs_normalized/BTC.csv",header=0)
print(csv.columns)

csv= csv.set_index("Date")
X= np.array(csv)
columns=csv.columns
y=csv['Close']
rf = RandomForestRegressor(n_jobs=-1, n_estimators=14, verbose=0)
rf.fit(X,y)
#print(rf.feature_importances_)
"""ranks["RF"] = ranking(rf.feature_importances_, columns)
print(ranks)"""
selector = RFE(rf, n_features_to_select=3, step=1)
selector = selector.fit(X,y)
"""print "Features sorted by their rank:"""
print (sorted(zip(map(lambda x: round(x, 4), selector.ranking_), columns)))

i=0
ranking={'symbol':[],'position':[]}
for col in csv.columns.values:
    ranking['symbol'].append(col)
    ranking['position'].append(selector.ranking_[i])
    i+=1
pd.DataFrame(data=ranking).to_csv("../main/prova1.csv",index=False)
"""print(selector.get_support())
print(selector.ranking_)"""

#selector.support_array([ True,  True,  True,  True,  True, False, False, False, False,False])
#selector.ranking_array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])





