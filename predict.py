# usage: python predict.py {database_host} {model_filepath} {output_filepath}
# e.g. python predict.py localhost:8080 ./model.h5 ./sample_output.csv

# 1) Read data from database
# 	input = table
# 2) Output CSV
# 	post_key: string type
# 	is_trending: bool type

f_score = FBeta(average='macro', beta=1)
learn = tabular_learner(dataLoader, layers=layers, emb_szs=emb_szs, metrics=f_score)
learn.load('best')

# learn = load_learner('model_path', 'trained_model.pkl', test=df)

l = []
for i in range(1000):
  a = learn.predict(df.iloc[i])
  l.append(a)