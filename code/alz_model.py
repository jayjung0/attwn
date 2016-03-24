import numpy as np
import scipy.stats as scs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import izip

class alz_model(object):
	def __init__(self):
		self.ss = StandardScaler()
		self.pca = PCA(n_components=5)
		self.X = None

	def _transform(self, X):
		return self.pca.transform(self.ss.transform(X))

	def fit(self, Xtrain):
		self.ss.fit(Xtrain)
		self.pca.fit(self.ss.transform(Xtrain))
		self.gkde = scs.gaussian_kde(self._transform(Xtrain)[:, :2].T)

	def predict(self, X):
		self.X = self._transform(X)[:, :2]
		return self.gkde(self.X.T)

	def predict_contour(self, X):
		return self.gkde(X) 

	def plot(self, X, train_size, birth_year, ind_splits, years, save_plot=False,\
			save_name='alz_plot', connect_scatter=False):
		'''
		Inputs are feature matrix X in chronological order, training size train_size,
		year of author's birth, list of indices to split for charts (for example,
		[0,5,10] will create 2 charts:  one from index 0:5, one from index 5:10),
		and year of each item's publication.
		
		Outputs are plots
		'''
		sns.set_style('dark')
		X_train = X[:train_size].copy()
		self.fit(X_train)
		X_transform = self._transform(X)
		xmin = X_transform[:, 0].min()-np.std(X_transform[:, 0])
		xmax = X_transform[:, 0].max()+np.std(X_transform[:, 0])
		ymin = X_transform[:, 1].min()-np.std(X_transform[:, 1])
		ymax = X_transform[:, 1].max()+np.std(X_transform[:, 1])
		XX, YY = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
		positions = np.vstack([XX.ravel(), YY.ravel()])
		Z = np.reshape(self.predict_contour(positions), XX.shape)
		t = years-birth_year

		for tup in izip(ind_splits[:-1], ind_splits[1:]):
			fig, ax = plt.subplots(figsize=(15, 9))
			plt.contour(XX, YY, Z, 8, cmap='Blues_r', alpha=.7)
			s = np.zeros(len(X_transform))
			s[tup[0]: tup[1]] = 50
			plt.scatter(X_transform[:, 0], X_transform[:, 1], s=s, cmap='Oranges', c=t)
			plt.colorbar()
			if connect_scatter:
				plt.plot(X_transform[:, 0], X_transform[:, 1], ls='-.')
			ax.set_xlim([xmin, xmax])
			ax.set_ylim([ymin, ymax])
			if save_plot:
				plt_name = save_name + '_%s_%s' % (str(tup[0]), str(tup[1]))
				plt.savefig(plt_name, bbox_inches='tight',dpi=300)#, pad_inches=1)
			else:
				plt.show()
