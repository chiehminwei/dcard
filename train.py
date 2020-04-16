# usage: python train.py {database_host} {model_filepath}
# e.g. python train.py localhost:8080 ./model.h5

import sys
import pandas as pd
from functools import reduce
import numpy as np

from utils import postgres_connector, train_queries, test_queries, add_datepart


def load_df(engine):
	print('Loading datasets from server...')
	dfs = []
	# temporarily left out for debugging
	# for query in train_queries:
	for query in test_queries:
	  dfs.append(pd.read_sql(query, engine))

	print('Datasets loaded. Joining on post_key...')
	
	df = reduce(lambda left,right: pd.merge(left,right,on='post_key'), dfs)
	df.drop('post_key', axis=1, inplace=True)
	
	print('Datsets joined.')
	print(df.info())

	print('Cleaning data...')
	# 為了簡化問題複雜度，我們目前訂為在文章發出的 36 小時內愛心數 >= 1000 就是熱門文章。
	df['is_trending'] = df['like_count_36_hour'] >= 1000
	df.is_trending = df.is_trending.astype(int)

	# Convert datetime field into categorical attributes
	add_datepart(df, 'created_at_hour', time=True)
	print('Datsets cleaned.')
	print(df.info())
	return df

def createDataLoader(df, cat_names, dep_var, path="model_path", sample_frac=0.1, dev_set_size=2000, procs=None):
	df = df.sample(frac=sample_frac).reset_index(drop=True)
	if not procs:
		procs = [FillMissing, Categorify, Normalize]
	valid_idx = range(len(df)-dev_set_size, len(df))
	data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
	return data

def train(dataLoader, layers, emb_szs, model_filepath, lr=5e-2,  num_epochs=20):
	learn = tabular_learner(dataLoader, layers=layers, emb_szs=emb_szs, metrics=accuracy, callback_fns=ShowGraph)
	# The next 2 lines gave the starting lr of 5e-2
	# learn.lr_find()
	# learn.recorder.plot() 
	learn.fit_one_cycle(num_epochs, lr)
	learn.save(model_filepath, return_path=True)

if __name__ == "__main__":

	database_host, model_filepath = None, None
	if len(sys.argv) > 2:
		database_host = sys.argv[2]
	if len(sys.argv) > 3:
		model_filepath = sys.argv[3]
	if not database_host:
		database_host = "35.187.144.113"
	if not model_filepath:
		model_filepath = "trained_model"

	# Get connect engine   
	engine = postgres_connector(
	   database_host,
	   5432,
	   "intern_task",
	   "candidate",
	   "dcard-data-intern-2020"
	)
	df = load_df(engine)
	
	from fastai.tabular import *
	cat_names = ['created_at_dayofweek',  'created_at_hour']
	dep_var = 'is_trending'
	path = 'model_path'
	
	print('Creating data loader...')
	dataLoader = createDataLoader(df, cat_names, dep_var, path)
	print('Data loader created.')

	layers=[200,100]
	emb_szs={'created_at_Dayofweek': 10, 'created_at_Hour': 15}
	print('Using a {}-level MLP. Sizes are '.format(len(layers)) + layers)
	print(emb_szs)
	train(dataLoader, layers, emb_szs, model_filepath, num_epochs=10)

