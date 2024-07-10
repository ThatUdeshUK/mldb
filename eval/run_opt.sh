cd ../../duckdb

export RUNS=1
export BATCH_SIZES=(1 4 16 64 256) # 2048 4096) #  4096
export ONNX_INTRA_THREADS=(8 16 32 64) # 64 128 256 512 1024 4096
export LLM_MAX_TOKENS_LIST=(128 256 384 512)
# export CARDINALITIES=(10000000) # 1000000 5000000  15000000 20000000
export CARDINALITY=1024
export LLM_MAX_TOKENS=512

for RUN in $( seq 1 $RUNS )
do
    for BATCH_SIZE in ${BATCH_SIZES[@]};
    do
        for INTRA_THREAD in ${ONNX_INTRA_THREADS[@]};
        do
            # for LLM_MAX_TOKENS in ${LLM_MAX_TOKENS_LIST[@]};
            # do
            # for CARDINALITY in ${CARDINALITIES[@]};
            # do
        echo "RUN -> $RUN, BATCH_SIZE -> $BATCH_SIZE, LLM_MAX_TOKENS -> $LLM_MAX_TOKENS, INTRA_TC -> $INTRA_THREAD, N -> $CARDINALITY."
./build/release/duckdb ../mldb/test_4096.db << QUERIES
PRAGMA enable_profiling='json';
SET ml_batch_size = $BATCH_SIZE;
SET onnx_intra_tc = $INTRA_THREAD;
SET llm_max_tokens = $LLM_MAX_TOKENS;
# PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/tabular_performance/opt_vec_onnx_dnn|7|1024_iris_n_${CARDINALITY}_bs_${BATCH_SIZE}_run_${RUN}_intratc_${INTRA_THREAD}.json';
PRAGMA profile_output='/homes/ukumaras/Projects/mldb/eval/llm_asc/opt_vec_onnx_distilbert_imdb_n_${CARDINALITY}_lmt_${LLM_MAX_TOKENS}_bs_${BATCH_SIZE}_run_${RUN}_intratc_${INTRA_THREAD}_asc_2_dyn_0.json';
# SELECT * FROM PREDICT ('/homes/ukumaras/Projects/mldb/models/iris_d7_w1024_out+class.onnx', (FROM iris_20_0 LIMIT ${CARDINALITY})) BY (sepal_length, sepal_width, petal_length, petal_width) WITH (setosa FLOAT, versicolour FLOAT, virginica FLOAT, cls FLOAT);
SELECT * FROM PREDICT ('/homes/ukumaras/scratch/Models/distilbert-base-uncased-finetuned-sst-2-english-onnx', (FROM (FROM imdb WHERE stage == 'test' LIMIT ${CARDINALITY}) ORDER BY length(text))) BY (text) WITH (neg FLOAT, pos FLOAT);
.q
QUERIES
            # done
        done
    done
done


