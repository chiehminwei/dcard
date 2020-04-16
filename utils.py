import sqlalchemy
import numpy as np
import pandas as pd

# Connector function
def postgres_connector(host, port, database, user, password=None):
   user_info = user if password is None else user + ':' + password
   # example: postgresql://federer:grandestslam@localhost:5432/tennis
   url = 'postgres://%s@%s:%d/%s' % (user_info, host, port, database)
   return sqlalchemy.create_engine(url, client_encoding='utf-8')

# Convert datetime field of dataframe into categorical attributes
def add_datepart(df, fldname, errors="raise"):
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