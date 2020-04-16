import pandas as pd
import sqlalchemy
from functools import reduce
import numpy as np

# Connector function
def postgres_connector(host, port, database, user, password=None):
   user_info = user if password is None else user + ':' + password
   # example: postgresql://federer:grandestslam@localhost:5432/tennis
   url = 'postgres://%s@%s:%d/%s' % (user_info, host, port, database)
   return sqlalchemy.create_engine(url, client_encoding='utf-8')
# Get connect engine   
engine = postgres_connector(
   "35.187.144.113",
   5432,
   "intern_task",
   "candidate",
   "dcard-data-intern-2020"
)

# Queries
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

queries = [
  posts_train_query,
  post_shared_train_query,
  post_comment_created_train_query,
  post_liked_train_query,
  post_collected_train_query
]
dfs = []
for query in queries:
  dfs.append(pd.read_sql(query, engine))

print('Datasets loaded from server.')

df = reduce(lambda left,right: pd.merge(left,right,on='post_key'), dfs)
df.drop('post_key', axis=1, inplace=True)

print('Datsets joined.')

# 為了簡化問題複雜度，我們目前訂為在文章發出的 36 小時內愛心數 >= 1000 就是熱門文章。
df['is_trending'] = df['like_count_36_hour'] >= 1000
df.is_trending = df.is_trending.astype(int)

def add_datepart(df, fldname, drop=True, time=False, errors="raise"):
    "Create many new columns based on datetime column."
    fld = df[fldname]
    fld_dtype = fld.dtype
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64
    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld,
                      infer_datetime_format=True, errors=errors)
    targ_pre = 'created_at_'
    attr = ['Dayofweek']
    if time: attr = attr + ['Hour']
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    if drop: df.drop(fldname, axis=1, inplace=True)

add_datepart(df, 'created_at_hour', time=True)

print('Datsets processed into dataframes.')


from fastai.tabular import *
procs = [FillMissing, Categorify, Normalize]
valid_idx = range(len(df)-2000, len(df))

cat_names = ['created_at_Dayofweek',  'created_at_Hour']
path = 'model_path'
dep_var = 'is_trending'

print('Creating fastai data loader...')
data = TabularDataBunch.from_df(path, df, dep_var, valid_idx=valid_idx, procs=procs, cat_names=cat_names)
print('Data loader created.')

learn = tabular_learner(data, layers=[200,100], emb_szs={'created_at_Dayofweek': 10, 'created_at_Hour': 15}, metrics=accuracy, callback_fns=ShowGraph)
learn.fit_one_cycle(50, 1e-2)

learn.save("trained_model", return_path=True)
