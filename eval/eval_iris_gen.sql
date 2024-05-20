PRAGMA enable_profiling='json';

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris3_1000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 1000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris3_5000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 5000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris3_10000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 10000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris3_15000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 15000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris3_20000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 20000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris4_1000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 1000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris4_5000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 5000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris4_10000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 10000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris4_15000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 15000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris4_20000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 20000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris5_1000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 1000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris5_5000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 5000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris5_10000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 10000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris5_15000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 15000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

PRAGMA profile_output='/Users/udeshuk/Developer/mldb/eval/analyze_outputs/opt_vec_onnx_iris5_20000000.json';
SELECT *  FROM PREDICT ('/Users/udeshuk/Developer/mldb/models/iris_d1_w128.onnx', (SELECT *  FROM iris_20_0 LIMIT 20000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT) as a;

