#!/usr/bin/env python
# coding: utf-8

# Export Iris Dataset to DuckDB

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import duckdb

table_name = 'iris'
ori_workload = False

if not ori_workload:
    workload = 20000000
    table_name += "_" + str(workload / 1000000).replace(".", "_")

iris = datasets.load_iris()
iris_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

scaler = StandardScaler()
iris_data = scaler.fit_transform(iris.data)

iris_data = pd.DataFrame(iris_data, columns=iris_cols)
iris_data['label'] = iris.target
iris_data.head()

if not ori_workload:
    iris_data = iris_data.sample(n=workload, replace=True)

# Connect to the DB

con = duckdb.connect("../test.db")
con.sql(f"DROP TABLE IF EXISTS {table_name}")
con.sql(f"CREATE TABLE {table_name} (sepal_length float, sepal_width float, petal_length float, petal_width float, label integer)")
con.sql(f"INSERT INTO {table_name} SELECT * FROM iris_data")
con.sql(f"SELECT * FROM {table_name}").show()
con.close()

# iris_data.to_csv('../out.csv', header=False, index=False)




