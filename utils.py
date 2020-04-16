import sqlalchemy
import numpy as np
import pandas as pd
from functools import reduce

# Connector function
def postgres_connector(host, port, database, user, password=None):
   user_info = user if password is None else user + ':' + password
   # example: postgresql://federer:grandestslam@localhost:5432/tennis
   url = 'postgres://%s@%s:%d/%s' % (user_info, host, port, database)
   return sqlalchemy.create_engine(url, client_encoding='utf-8')

# Convert datetime field of dataframe into categorical attributes
def my_add_datepart(df, fldname, errors="raise"):
    "Create many new columns based on datetime column."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld,
                      infer_datetime_format=True, errors=errors)
    df['created_at_dayofweek'] = fld.dt.dayofweek
    df['created_at_hour'] = fld.dt.hour
    
def load_df(engine, mode='train'):
  print('Loading datasets from server...')
  dfs = []
  # temporarily left out for debugging
  if mode == 'train':
    for query in train_queries:
      dfs.append(pd.read_sql(query, engine))
  elif mode == 'debug':
      dfs.append(pd.read_sql(debug_query, engine))
  elif mode == 'debug_pred':
      dfs.append(pd.read_sql(debug_pred_query, engine))
  else:
    for query in test_queries:
      dfs.append(pd.read_sql(query, engine))

  print('Datasets loaded. Joining on post_key...')
  
  df = reduce(lambda left,right: pd.merge(left,right,on='post_key'), dfs)
  
  print('Datasets joined.')
  print(df.info())

  print('Cleaning data...')
  if mode == 'train':
    sample_frac = 0.5
    print('Sample {} fraction of data to avoid OOM issues.'.format(sample_frac))
    df = df.sample(frac=sample_frac).reset_index(drop=True)
  # 為了簡化問題複雜度，我們目前訂為在文章發出的 36 小時內愛心數 >= 1000 就是熱門文章。
  if mode == 'train' or mode == 'debug':
    df['is_trending'] = df['like_count_36_hour'] >= 1000
    df.is_trending = df.is_trending.astype(int)
    df.drop('like_count_36_hour', axis=1, inplace=True)

  # Convert datetime field into categorical attributes
  my_add_datepart(df, 'created_at_hour')
  print('Datsets cleaned.')
  print(df.info())
  return df

# Queries
debug_query = """
SELECT *
FROM posts_train
LIMIT 5000
"""

debug_pred_query = """
SELECT *
FROM posts_test
LIMIT 5000
"""

posts_train_query = """
SELECT *
FROM posts_train
"""

post_shared_train_query = """
SELECT post_key, count AS share_count
FROM post_shared_train
"""

post_comment_created_train_query = """
SELECT post_key, count AS comment_count
FROM post_comment_created_train
"""

post_liked_train_query = """
SELECT post_key, count AS like_count
FROM post_liked_train
"""

post_collected_train_query = """
SELECT post_key, count AS collect_count
FROM post_collected_train
"""

###
posts_test_query = """
SELECT *
FROM posts_test
"""

post_shared_test_query = """
SELECT post_key, count AS share_count
FROM post_shared_test
"""

post_comment_created_test_query = """
SELECT post_key, count AS comment_count
FROM post_comment_created_test
"""

post_liked_test_query = """
SELECT post_key, count AS like_count
FROM post_liked_test
"""

post_collected_test_query = """
SELECT post_key, count AS collect_count
FROM post_collected_test
"""

train_queries = [
  posts_train_query,
  post_shared_train_query,
  post_comment_created_train_query,
  post_liked_train_query,
  post_collected_train_query
]

test_queries = [
  posts_test_query,
  post_shared_test_query,
  post_comment_created_test_query,
  post_liked_test_query,
  post_collected_test_query
]