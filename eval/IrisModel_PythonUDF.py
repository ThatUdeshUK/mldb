# %% [markdown]
# # Evaluate Python UDF Performance (Iris)

# %%
import pandas as pd

import time
import json

import duckdb
from duckdb.typing import *

import torch
import pyarrow as pa

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("workload")
args = parser.parse_args()

# %% [markdown]
# ### Config

# %%
times = {}

# %%
table_name = 'iris'
times['dataset'] = 'iris'
times['pipeline'] = 'python_udf_vec'

drop_table = False
workload = args.workload

# %% [markdown]
# ### Load iris data

# %%
times['workload'] = workload

# %% [markdown]
# ### Load the compiled model

# %%
con = duckdb.connect("../test.db")

# %%
def load_iris():
    model_path = f"{args.path}/models/iris_d1_w128.pt"
    load_iris.model = torch.jit.load(model_path)
    return True

load_iris.model = None

st = time.perf_counter_ns()

con.create_function("load_iris", load_iris, [], BOOLEAN, side_effects=True)
res = con.sql("SELECT load_iris()").show()

et = time.perf_counter_ns()
times["load"] = (et - st)/1000

# %% [markdown]
# ### Predict

# %%
# @udf.scalar.pyarrow
# def predict_fare(x: dt.float64) -> dt.float32:
def predict_iris(a, b, c, d):
    a = torch.from_numpy(a.to_numpy()[:, None]).float()
    b = torch.from_numpy(b.to_numpy()[:, None]).float()
    c = torch.from_numpy(c.to_numpy()[:, None]).float()
    d = torch.from_numpy(d.to_numpy()[:, None]).float()

    # print(a.shape)
    
    # x = torch.Tensor([a, b, c, d])
    x = torch.cat([a, b, c, d], -1)
    # print(x)

    out = load_iris.model(x).detach().argmax(dim=1)

    # print(out)
    
    return pa.array(out.numpy())

con.sql("PRAGMA enable_profiling='json'")
con.create_function("predict_iris", predict_iris, [FLOAT, FLOAT, FLOAT, FLOAT], FLOAT, side_effects=True, type='arrow')
res = con.sql(f"EXPLAIN ANALYZE SELECT *, predict_iris(sepal_length, sepal_width, sepal_width, sepal_width) FROM iris_20_0 LIMIT {workload}").fetchall()

con.close()

# %%
prediction = json.loads(res[0][1])['children'][0]['children'][0]['children'][0]['children'][0]
times["predict"] = prediction['timing'] * 1000000

move = prediction['children'][0]
times["move"] = move['timing'] * 1000000

# %%
times_df = pd.DataFrame.from_records([times]).loc[:, ['dataset', 'pipeline', 'workload', 'move', 'load', 'predict']]

con = duckdb.connect("../test.db")
tables = con.sql(f"SHOW TABLES").df()
has_table = tables[tables.name == 'times'].shape[0] == 1

if drop_table:
    con.sql(f"DROP TABLE IF EXISTS times")
    
if not has_table:
    con.sql(f"CREATE TABLE times (dataset varchar, workload integer, move integer, load integer, predict integer, ts timestamp DEFAULT current_timestamp)")

con.sql(f"INSERT INTO times (dataset, pipeline, workload, move, load, predict) SELECT * FROM times_df")
con.sql(f"SELECT * FROM times").show()
con.close()

# %%



