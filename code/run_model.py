import alz_model
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import scipy.stats as scs
import seaborn as sns

db_in = 'ac_features'
num_train = 30

columns = ['indef_words_pct','3_grams_avg','4_grams_avg','5_grams_avg',\
			'unique_verbs_pct', 'adj_verb_ratio']

model = alz_model.alz_model()
engine = create_engine('postgresql://jayjung@localhost:5432/alz')
df_in = pd.read_sql_query('select * from ' + db_in + ';', con=engine)
df_in.sort('year', inplace=True)
X = df_in[columns].values

model.plot(X, num_train, 1890, [0, 73], df_in.year.values)