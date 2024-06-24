PRAGMA enable_profiling='json';

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris3_1000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 1000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris3_5000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 5000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris3_10000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 10000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris3_15000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 15000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris3_20000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 20000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris4_1000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 1000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris4_5000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 5000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris4_10000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 10000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris4_15000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 15000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris4_20000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 20000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris5_1000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 1000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris5_5000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 5000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris5_10000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 10000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris5_15000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 15000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/analyze_outputs/opt_vec_onnx_out+class_row_row_basic_iris5_20000000.json';
SELECT *  FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d1_w128_out+class.onnx', (SELECT sepal_length, sepal_width, petal_length, petal_width FROM iris_20_0 LIMIT 20000000)) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT) as a;

