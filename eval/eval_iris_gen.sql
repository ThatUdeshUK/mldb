PRAGMA enable_profiling='json';

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_col_copy_iris5_20000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.pt', (SELECT *  FROM iris_20_0 LIMIT 20000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

