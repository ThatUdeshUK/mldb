# Relational ML operator experiments

Evaluate the performance behaviors of the relational ML operator on varying model in cluding DNNs, language models, and GNNs.

## Directory structure

```
<directory>
│   README.md    
│
└───eval
│   │   Compare *.ipynb - Notebooks to compare and visualize results
│   │   <dataset>Model_<method>.ipynb - Notebooks to run experiments
│   │   run_opt.sh - Running experiments for rel. ML operator (lm and tabular)
│   
└───finetune
│   │   FinetuneBertModel.ipynb - Notebooks to finetuning BERT models
│
└───models and train models
│   │   Export*.ipynb - Notebooks to ingest data into duckdb
│   │   Train*.ipynb - Notebooks to train ONNX models
│
└───tsl - <under development>
```
