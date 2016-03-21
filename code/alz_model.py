import pandas as pd 
import numpy as np 
import scipy.stats as scs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class alz_model(object):
	def __init__(self):
		self.ss = StandardScaler()
		self.pca = PCA(n_components=5)
		self.X = None

	def _transform(self, X):
		return self.pca.transform(self.ss.transform(X))

	def fit(self, Xtrain):
		self.ss.fit(Xtrain)
		self.pca.fit(Xtrain)
		self.gkde = scs.gaussian_kde(self.pca.transform(Xtrain)[:, :2].T)

	def predict(self, X):
		self.X = self._transform(X)[:, :2]
		return self.gkde(self.X.T)

	def predict_contour(self, X):
		return self.gkde(X.T)

	# def plot_scree(self):

# db_in = 'rm_features'
# columns = ['unique_words_pct', 'indef_words_pct', 'word_counts_avg',\
# 'adj_pct', 'verb_pct', 'adj_verb_ratio', 'adj_noun_ratio', \
# 'unique_verbs_pct', '2_grams_avg', '3_grams_avg', '4_grams_avg', '5_grams_avg']

# model = alz_model()
# engine = create_engine('postgresql://jayjung@localhost:5432/alz')
# df_in = pd.read_sql_query('select * from ' + db_in + ';', con=engine)
# X = df_in[columns].values
# Xtrain = X[:9].copy()
# model.fit(Xtrain)
# p = model.predict(X)
