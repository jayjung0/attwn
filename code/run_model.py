import alz_model
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import scipy.stats as scs
import seaborn as sns

db_in = 'ac_features'
num_train = 30

columns = ['unique_words_pct', 'indef_words_pct', \
'2_grams_avg', '3_grams_avg', '4_grams_avg']#,'adj_noun_ratio', \
# 'adj_pct',  '5_grams_avg', 'verb_pct', 'adj_verb_ratio', 'unique_verbs_pct','word_counts_avg']

model = alz_model.alz_model()
engine = create_engine('postgresql://jayjung@localhost:5432/alz')
df_in = pd.read_sql_query('select * from ' + db_in + ';', con=engine)
df_in.sort('year', inplace=True)
X = df_in[columns].values
Xtrain = X[:num_train].copy()
model.fit(Xtrain)
p = model.predict(X)


def estimate_gaussian(X):
    m, n = X.shape
    mu = np.mean(X,axis=0)
    cov = np.cov(X.T)
    return mu, cov
mu, cov = estimate_gaussian(model.X[:num_train,:2])
g=sns.jointplot(model.X[:num_train,0],model.X[:num_train,1], kind="kde", size=10)
g.ax_joint.scatter(model.X[:,0],model.X[:,1],c='r')

for i, txt in enumerate(np.arange(len(model.X))):
    g.ax_joint.annotate(txt, (model.X[i,0],model.X[i,1]))
plt.show()

# #make plots
# xmin = model.X[:,0].min()-np.std(model.X[:,0])
# xmax = model.X[:,0].max()+np.std(model.X[:,0])
# ymin = model.X[:,1].min()-np.std(model.X[:,1])
# ymax = model.X[:,1].max()+np.std(model.X[:,1])
# print xmin, xmax, ymin, ymax
# XX, YY = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([XX.ravel(), YY.ravel()])
# Z = np.reshape(model.predict_contour(positions.T).T, XX.shape)
# fig, ax = plt.subplots()
# # plt.contourf(X,Y,Z,8, cmap='Blues',alpha=.7)
# plt.contour(XX, YY, Z, 8, cmap='winter', alpha=.7)

# ax.plot(model.X[:,0], model.X[:,1], 'k.', markersize=10)
# # ax.set_xlim([xmin, xmax])
# # ax.set_ylim([ymin, ymax])
# n_drops = len(model.X)
# rain_drops = np.zeros(n_drops, dtype=[('position', float, 2),
#                                       ('size',     float, 1),
#                                       ('color',    float, 4)])
# rain_drops['position'] = model.X[:,:2]

# scat = ax.scatter(rain_drops['position'][:, 0], rain_drops['position'][:, 1],
#                   s=rain_drops['size'], lw=.5, edgecolors=rain_drops['color'],
#                   facecolors='none')
# for i, txt in enumerate(np.arange(len(model.X))):
#     ax.annotate(txt, (model.X[i,0],model.X[i,1]))

# plt.show()