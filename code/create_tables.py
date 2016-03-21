import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
from collections import Counter
import spacy
import json

def create_parsed_books_table(db_in, db_out):
	'''
	Creates new database in postgres with parsed books
	Input: db_in: string.  db containing name, year, and booktext for author
		db_out:  string.  name of db to be created with books parsed
	'''
	engine = create_engine('postgresql://jayjung@localhost:5432/alz')
	df_in = pd.read_sql_query('select * from ' + db_in + ';', con=engine)
	df = df_in[['name', 'year']].copy()
	print 'step1'
	make_nltk_features(df, df_in['book_text'], [words, unique_words, indef_dict, word_lengths], tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize)
	print 'step2'
	make_nltk_features(df, df_in['book_text'], [sentence_length])
	print 'step3'
	make_nltk_features(df, df_in['book_text'], [word_counts, n_grams], tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+').tokenize, \
	 lemmatizer=nltk.wordnet.WordNetLemmatizer())
	print 'step4'
	make_POS_features(df, df_in['book_text'], [verb_counts, adj_counts])
	df.to_sql(db_out, engine, if_exists='replace', chunksize=20000)

def make_nltk_features(df, books, feat_names, tokenizer=None, grouping=None, lemmatizer=None):
	# adds features to dataframe.  can indicate use of tokenizer and lemmatizer.  lemmatizing requires tokenizing
	# save runtime by grouping features by tokenizer and lemmatizer
	feat_list = [[] for feat in feat_names]
	for book in books:
		if tokenizer:
			book = tokenizer(book)
			if lemmatizer:
				book = map(lambda y: lemmatizer.lemmatize(y, 'v'), map(lambda x: lemmatizer.lemmatize(x.lower()), book))
		for i, feat in enumerate(feat_names):
			feat_list[i].append(feat(book))
	for i, feat in enumerate(feat_names):
		df[feat.__name__] = feat_list[i]

def make_nltk_features2(df, books, feat_names, tokenizer=None, grouping=None, lemmatizer=None):
	for feat in feat_names:
		df[feat.__name__] = books

def make_POS_features(df, books, feat_names):
	# make Pos features using spacy
	all_books_pos_list = []
	nlp = spacy.en.English()
	pos_counter = []
	for book in books:
		print 'step4-'
		book = nlp(book)
		book_pos_list = []
		for token in book:
			book_pos_list.append([token.lemma_, token.pos_])
		all_books_pos_list.append(book_pos_list)
		pos_counter.append(json.dumps(Counter(np.array(book_pos_list)[:,1])))
	for feat in feat_names:
		#for each feature, get run each feature in each book, append to df
		lst = []
		for pos_list in all_books_pos_list:
			lst.append(feat(pos_list))
		df[feat.__name__] = lst
	df['pos_counts'] = pos_counter

def verb_counts(lst):
	lst = np.array(lst)
	return json.dumps(Counter(lst[lst[:,1] == 'VERB'][:,0]))

def adj_counts(lst):
	lst = np.array(lst)
	return json.dumps(Counter(lst[lst[:,1] == 'ADJ'][:,0]))

def words(lst):
	# returns number of words in book
	return len(lst)

def unique_words(lst):
	# returns number of unique words in a book
	return len(set(lst))

def indef_dict(lst):
	# returns dictionary of indefite words and their count
	myDict = {}
	indefinite_words = ('anybody','anyone','anything','everybody','everyone','everything','nobody','none','nothing','somebody',\
		'someone','something','all','another','any','each','either','few','many','neither','one','some','several')
	myDict = myDict.fromkeys(indefinite_words, 0)
	for word in lst:
		if word.lower() in myDict:
			myDict[word.lower()] += 1
	return json.dumps(myDict)

def word_counts(lst):
	# returns dictionary of words and number of times used
	return json.dumps(Counter(lst))

def word_lengths(lst):
	return json.dumps(map(len, lst))

def sentence_length(s):
	s = re.sub('\.+', '.', s)
	s = s.split('.')
	lengths = []
	for sentence in s:
		lengths.append(len(sentence.split()))
	return json.dumps(lengths)

def n_grams(lst):
	myLst = []
	for i in xrange(2, 6):
		ng = nltk.ngrams(lst, i)
		fdist = nltk.FreqDist(ng)
		myLst.append(fdist.values())
	return json.dumps(myLst)

# create_parsed_books_table('ac_books', 'ac_parsed_books')
create_parsed_books_table('sk_books', 'sk_parsed_books')