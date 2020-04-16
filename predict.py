# usage: python predict.py {database_host} {model_filepath} {output_filepath}
# e.g. python predict.py localhost:8080 ./model.h5 ./sample_output.csv
# python predict.py 35.187.144.113 models/trained_model.pkl predictions.csv

# 1) Read data from database
# 	input = table
# 2) Output CSV
# 	post_key: string type
# 	is_trending: bool type

from sklearn.metrics import f1_score
from fastai.tabular import *
from utils import postgres_connector, load_df
import os

database_host, model_filepath = None, None
if len(sys.argv) != 4:
	print('usage: python predict.py {database_host} {model_filepath} {output_filepath}')
	print('e.g. python predict.py localhost:8080 ./model.h5 ./sample_output.csv')

database_host = sys.argv[1]
model_filepath = sys.argv[2]
output_filepath = sys.argv[3]

file_path = '/'.join(model_filepath.split('/')[:-1])
file_name = model_filepath.split('/')[-1]

output_file_path = '/'.join(output_filepath.split('/')[:-1])
output_file_name = output_filepath.split('/')[-1]
try:
	os.mkdir(output_file_path)
except OSError:
	print('Failed to create output directory.')
else:
	print('Succesfully created directory.')

# Get connect engine   
engine = postgres_connector(
   database_host,
   5432,
   "intern_task",
   "candidate",
   "dcard-data-intern-2020"
)
df = load_df(engine, mode='test')
learn = load_learner(file_path, file_name, test=TabularList.from_df(df))
preds = learn.get_preds(ds_type=DatasetType.Test)[1].numpy()
final_df = pd.DataFrame({'post_key': df['post_key'], 'is_trending': preds})
final_df.to_csv('predictions.csv', header=True, index=False)


df['is_trending'] = df['like_count_36_hour'] >= 1000
df.is_trending = df.is_trending.astype(int)
y_true = df['is_trending'].to_numpy()
y_pred = preds
print('f1_socre:')
print(f1_score(y_true, y_pred, average='macro'))
