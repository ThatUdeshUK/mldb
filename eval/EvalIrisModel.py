#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

import time

import torch
import torch.nn as nn

import duckdb


times = {}
table_name = 'iris'
times['dataset'] = 'iris'


ori_workload = False
drop_table = False

if not ori_workload:
    workload = 20000000
    table_name += "_" + str(workload / 1000000).replace(".", "_")
    select_stmt = f"SELECT * FROM iris_20_0 LIMIT {workload}"
else:
    select_stmt = f"SELECT * FROM iris"


st = time.perf_counter_ns()
con = duckdb.connect("../test.db")
iris = con.sql(select_stmt).df()
con.close()
et = time.perf_counter_ns()
times['workload'] = iris.shape[0]
times["move"] = (et - st)/1000


st = time.perf_counter_ns()
model_path = "../models/iris.pt"
model = torch.jit.load(model_path)
et = time.perf_counter_ns()
times["load"] = (et - st)/1000


st = time.perf_counter_ns()
iris_data = iris.iloc[:, :4]
iris_label = iris.iloc[:, 4]
x = torch.FloatTensor(iris_data.values)
out = model(x)
et = time.perf_counter_ns()
times["predict"] = (et - st)/1000

times_df = pd.DataFrame.from_records([times]).loc[:, ['dataset', 'workload', 'move', 'load', 'predict']]

con = duckdb.connect("../test.db")
tables = con.sql(f"SHOW TABLES").df()
has_table = tables[tables.name == 'times'].shape[0] == 1

if drop_table:
    con.sql(f"DROP TABLE IF EXISTS times")
    
if not has_table:
    con.sql(f"CREATE TABLE times (dataset varchar, workload integer, move integer, load integer, predict integer, ts timestamp DEFAULT current_timestamp)")

con.sql(f"INSERT INTO times (dataset, workload, move, load, predict) SELECT * FROM times_df")
con.sql(f"SELECT * FROM times").show()
con.close()
