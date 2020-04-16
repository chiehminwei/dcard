# usage: python predict.py {database_host} {model_filepath} {output_filepath}
# e.g. python predict.py localhost:8080 ./model.h5 ./sample_output.csv

# 1) Read data from database
# 	input = table
# 2) Output CSV
# 	post_key: string type
# 	is_trending: bool type