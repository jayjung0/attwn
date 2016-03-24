import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import json
import sys

def create_all_features(db_in, db_out, engine_URL):
	engine = create_engine(engine_URL)
	df_in = pd.read_sql_query('select * from ' + db_in + ';', con=engine)
	df = df_in[['name', 'year']].copy()
	convert_json(df_in)
	df['unique_words_pct'] = df_in.unique_words / df_in.words
	df['indef_words_pct'] = [sum(d.values()) for d in df_in.indef_dict]/df_in.words
	df['word_counts_avg'] = map(lambda x: np.mean(x.values()), df_in.word_counts)
	df['adj_pct'] = get_pos_pct(df_in, 'ADJ')
	df['verb_pct'] = get_pos_pct(df_in, 'VERB')
	df['adj_verb_ratio'] = np.divide(get_pos_pct(df_in, 'ADJ'), get_pos_pct(df_in, 'VERB'))
	df['adj_noun_ratio'] = np.divide(get_pos_pct(df_in, 'ADJ'), get_pos_pct(df_in, 'NOUN'))
	df['unique_verbs_pct'] = [len(d)/float(sum(d.values())) for d in df_in.verb_counts]

	for i in xrange(2, 6):
		s = str(i) + '_grams_avg'
		df[s] = [np.mean(ngrams[i-2]) for ngrams in df_in['n_grams']]
	df.to_sql(db_out, engine, if_exists='replace', chunksize=20000)

def convert_json(df_in):
	columns_to_convert = ['indef_dict', 'word_lengths', 'sentence_length',\
	'word_counts', 'n_grams', 'verb_counts', 'adj_counts', 'pos_counts']
	for col in columns_to_convert:
		lst = [json.loads(x) for x in df_in[col]]
		df_in[col] = lst

def get_pos_pct(df_in, pos):
	return [(dct[pos] / float(sum(dct.values())-dct['SPACE']-dct.get('X', 0)-\
		dct.get('SYM', 0)-dct['PUNCT'])) for dct in df_in.pos_counts]

if __name__ == '__main__':
	'''
	Inputs:  1 - database containing title, year, and texts of books
			 2 - name of output database in postgres
			 3 - URL to connect to database.  for example: 'postgresql://jayjung@localhost:5432/alz'
	Outputs:  postgres database
	'''
	create_all_features(sys.argv[1], sys.argv[2], sys.argv[3])
