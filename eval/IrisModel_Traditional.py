# %% [markdown]
# # Evaluate Traditional Pipeline Performance (Iris)

# %%
import pandas as pd

import time

import torch
import duckdb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("workload")
args = parser.parse_args()

# %% [markdown]
# ### Config

# %%
times = {}

# %%
table_name = 'iris'
times['dataset'] = 'iris'
times['pipeline'] = 'traditional'

drop_table = False
workload = args.workload
select_stmt = f"SELECT * FROM iris_20_0 LIMIT {workload}"

# %% [markdown]
# ### Load iris data
print("START! " + args.workload)

# %%
st = time.perf_counter_ns()

con = duckdb.connect("../test.db")
iris = con.sql(select_stmt).df()
iris.to_csv('dummy.csv', index=False)
con.close()

iris = pd.read_csv('dummy.csv')

et = time.perf_counter_ns()
times['workload'] = iris.shape[0]
times["move"] = (et - st)/1000
print("Move Done!")

# %% [markdown]
# ### Load the compiled model

# %%
st = time.perf_counter_ns()

model_path = "../models/iris_d1_w128.pt"
model = torch.jit.load(model_path)

et = time.perf_counter_ns()
times["load"] = (et - st)/1000
print("Load Done!")

# %% [markdown]
# ### Predict

# %%
st = time.perf_counter_ns()

iris_data = iris.iloc[:, :4]

x = torch.FloatTensor(iris_data.values)

out = model(x)

et = time.perf_counter_ns()
times["predict"] = (et - st)/1000
print("Predict Done!")

# %%
times_df = pd.DataFrame.from_records([times]).loc[:, ['dataset', 'workload', 'pipeline', 'move', 'load', 'predict']]

con = duckdb.connect("../test.db")
tables = con.sql(f"SHOW TABLES").df()
has_table = tables[tables.name == 'times'].shape[0] == 1

if drop_table:
    con.sql(f"DROP TABLE IF EXISTS times")
    
if not has_table:
    con.sql(f"CREATE TABLE times (dataset varchar, workload integer, pipeline varchar, move integer, load integer, predict integer, ts timestamp DEFAULT current_timestamp)")

con.sql(f"INSERT INTO times (dataset, workload, pipeline, move, load, predict) SELECT * FROM times_df")
con.sql(f"SELECT * FROM times").show()
con.close()
print("Finished! " + args.workload)
