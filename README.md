# Sample-Efficient Surrogate Modeling for RAG Configuration Performance Prediction

A research project that predicts the performance of Retrieval-Augmented
Generation (RAG) configurations using surrogate machine-learning models.
The objective is to reduce expensive end-to-end RAG evaluations when
exploring a large configuration space.

## Table of Contents

-   [Overview](#overview)
-   [Motivation](#motivation)
-   [System Workflow](#system-workflow)
-   [RAG Configuration Space](#rag-configuration-space)
-   [Dataset and Experimental Setup](#dataset-and-experimental-setup)
-   [Surrogate Models](#surrogate-models)
-   [Evaluation Metrics](#evaluation-metrics)
-   [Configuration Selection](#configuration-selection)
-   [Repeated-Run Experiment](#repeated-run-experiment)
-   [Technology Stack](#technology-stack)
-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Running the Project](#running-the-project)
-   [Important Files](#important-files)
-   [Reproducibility](#reproducibility)
-   [Limitations](#limitations)
-   [Future Work](#future-work)

## Overview

Retrieval-Augmented Generation combines information retrieval with a
Large Language Model (LLM). A retriever finds relevant documents for a
query, and the retrieved content is supplied to an LLM as context for
answer generation.

RAG performance depends on several configuration parameters.
Exhaustively testing every combination requires many costly pipeline
executions. This project evaluates a selected set of configurations,
builds a configuration-performance dataset, and trains surrogate models
to predict the performance of unseen configurations.

**Main objective:** predict answer quality, latency, and token usage for
unseen RAG configurations while reducing the number of actual RAG
evaluations.

## Motivation

Different retrieval, chunking, filtering, reranking, and context-budget
settings can produce different quality and efficiency outcomes. Running
the complete RAG pipeline for every possible combination is expensive. A
surrogate model learns the relationship between configuration parameters
and measured outcomes from previously evaluated configurations, then
estimates outcomes for new configurations much more cheaply.

## System Workflow

``` text
HotpotQA Questions + Document Corpus
                 |
                 v
       Define RAG Configuration Space
                 |
                 v
      Select Configurations to Evaluate
                 |
                 v
          Run the Actual RAG Pipeline
                 |
                 v
       Measure Quality, Latency, Tokens
                 |
                 v
      Configuration-Performance Dataset
                 |
                 v
        Train Five Surrogate Models
                 |
                 v
       Evaluate on Unseen Configurations
                 |
                 v
       Compare Models and Data Budgets
                 |
                 v
       Configuration-Selection Experiments
```

For each question, the RAG pipeline chunks corpus documents, retrieves
candidate chunks, applies similarity filtering, optionally reranks
results, constructs context within a token budget, generates an answer
using the local LLM, and evaluates the answer against the reference.

## RAG Configuration Space

The experiments vary these features:

  -----------------------------------------------------------------------
  Parameter                           Description
  ----------------------------------- -----------------------------------
  `chunk_size`                        Number of tokens in each document
                                      chunk

  `chunk_overlap`                     Overlap between consecutive chunks

  `top_k`                             Number of final retrieved chunks
                                      used

  `similarity_threshold`              Minimum normalized similarity score
                                      for filtering

  `context_budget`                    Maximum token budget for retrieved
                                      context

  `retriever`                         Retrieval method: BM25 or dense

  `reranker`                          Whether the reranking stage is
                                      enabled
  -----------------------------------------------------------------------

The RAG runner may also use `retrieval_k`, the number of candidate
documents retrieved before filtering/reranking.

Example configuration:

``` json
{
  "chunk_size": 256,
  "chunk_overlap": 32,
  "retriever": "dense",
  "top_k": 5,
  "retrieval_k": 20,
  "similarity_threshold": 0.5,
  "context_budget": 1024,
  "reranker": false
}
```

`chunk_overlap` must be smaller than `chunk_size`. Use only values and
encodings supported by the project's configuration-generation and
preprocessing code.

## Dataset and Experimental Setup

-   **Dataset:** HotpotQA
-   **Questions:** 100
-   **Corpus:** 997 documents
-   **Valid configuration space:** 12,000 configurations
-   **Actually evaluated:** 400 configurations
-   **Training configurations:** 320
-   **Fixed held-out test configurations:** 80

SciFact was not included in the final reported experiment; the final
experiment dataset is HotpotQA.

The 400 evaluated configurations were selected using balanced
randomized, LHS-style sampling over the discrete configuration space.
The 320 training configurations are used for model training and
sample-efficiency experiments. The same 80 held-out configurations are
used for testing generalization to unseen configurations.

### Prediction targets

  Target                Meaning                      Better direction
  --------------------- ---------------------------- ------------------
  `mean_answer_f1`      Mean token-level answer F1   Higher
  `mean_answer_em`      Mean Exact Match             Higher
  `mean_latency`        Mean measured RAG latency    Lower
  `mean_total_tokens`   Mean total token usage       Lower

Additional retrieval/context statistics may be present in the experiment
outputs.

## Surrogate Models

Five regression approaches are compared:

1.  **Linear Regression** --- baseline linear relationship.
2.  **Polynomial Regression** --- models nonlinear relationships using
    polynomial features.
3.  **Random Forest Regression** --- tree-ensemble model that captures
    nonlinearities and feature interactions.
4.  **RBF Regression / RBF Network** --- uses radial basis functions to
    model nonlinear relationships.
5.  **Gaussian Process Regression** --- probabilistic regression
    approach for nonlinear relationships.

Each model predicts four targets: F1, EM, latency, and total tokens.
This gives 20 model-target predictors (5 models × 4 targets).

## Evaluation Metrics

  -----------------------------------------------------------------------
  Metric                  Meaning                 Preferred value
  ----------------------- ----------------------- -----------------------
  MAE                     Mean absolute           Lower
                          prediction error        

  RMSE                    Root mean squared       Lower
                          prediction error        

  R²                      Variance explained by   Higher
                          the model               

  Spearman correlation    Agreement between       Higher
                          predicted and actual    
                          rankings                

  Top-k overlap           Overlap between         Higher
                          predicted and actual    
                          top-k configurations    

  Regret                  Gap from the best       Lower
                          actual configuration    
                          under an objective      

  Training time           Time to fit the model   Lower

  Prediction time         Time to generate        Lower
                          predictions             
  -----------------------------------------------------------------------

Both prediction accuracy and ranking quality matter: a model may predict
values reasonably well but still rank the best configurations poorly.

## Sample-Efficiency Experiments

Training budgets:

``` text
20, 40, 60, 80, 100, 150, 200, 250, 320
```

For each budget, a subset of the 320 training configurations is used to
train all five models. Each model predicts the same fixed 80 held-out
configurations, and the evaluation metrics are calculated. The subsets
are nested and use a fixed seed (`42`), so larger budgets build on
smaller ones.

Total comparisons:

``` text
9 budgets × 5 models × 4 targets = 180 model-target-budget evaluations
```

This experiment measures how prediction accuracy and ranking quality
change as more evaluated configurations become available.

## Configuration Selection

A separate experiment examines whether surrogate predictions can help
select promising configurations from a pool of 1,000 unevaluated
candidates.

Strategies compared: - **Random Search:** selects configurations
randomly. - **LHS:** selects configurations intended to spread coverage
across the parameter space. - **Surrogate-guided selection:** ranks
candidates using surrogate predictions and selects promising
configurations for the relevant objective.

Selected configurations are evaluated using the actual RAG pipeline.
Objectives are to maximize F1, maximize EM, minimize latency, and
minimize total tokens. Results vary by target and model; no single
surrogate should be assumed to win every objective.

## Repeated-Run Experiment

A repeated-run experiment examines the stability of actual RAG
measurements: - 5 selected configurations - 5 repeats per
configuration - 100 questions per repeat - 25 configuration-repeat
runs - 2,500 question-level records

This measures variability in actual RAG outcomes; it is not a repeated
training experiment for the surrogate models.

## Technology Stack

-   Python
-   Ollama for local LLM serving
-   Qwen 2.5 3B (`qwen2.5:3b`) for the later local answer-generation
    setup
-   OpenAI-compatible Python client connected to the local Ollama
    endpoint
-   scikit-learn
-   NumPy and pandas
-   Sentence-transformers / embedding model for dense retrieval, as
    configured in the code
-   BM25 retrieval
-   Optional cross-encoder reranking
-   Matplotlib
-   Git and Git LFS where configured

## Prerequisites

1.  Python installed (prefer the version specified by the repository's
    dependency files).
2.  Git installed.
3.  Ollama installed and running locally.
4.  Required local LLM model downloaded.
5.  Project Python dependencies installed.
6.  HotpotQA question and corpus files available in the expected paths.
7.  Sufficient RAM and compute for embedding-model loading and RAG
    evaluation. A compatible GPU may accelerate embedding inference.

Install Ollama from https://ollama.com/ and download the model:

``` bash
ollama pull qwen2.5:3b
ollama list
```

Start the Ollama service if it is not already running:

``` bash
ollama serve
```

The local OpenAI-compatible endpoint used by the RAG LLM client is:

``` text
http://localhost:11434/v1
```

The API key used by the local Ollama-compatible client is a placeholder,
not an external API credential.

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/Kinshukk098/rag-surrogate-optimization.git
cd rag-surrogate-optimization
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows Git Bash:**

``` bash
python -m venv .venv
source .venv/Scripts/activate
```

**Linux / macOS:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If the repository contains `requirements.txt`:

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the repository uses another dependency manager or manifest, follow
that file instead. Do not replace pinned project dependencies with an
assumed package list.

### 4. Verify the local LLM

In a separate terminal:

``` bash
ollama list
curl http://localhost:11434/api/tags
curl http://localhost:11434/v1/models
```

Confirm that `qwen2.5:3b` is available and that the local endpoint
responds.

### 5. Check `experiment_config.json`

For the local setup described here, the LLM model should be set to:

``` json
"llm": {
  "model": "qwen2.5:3b",
  "temperature": 0,
  "max_completion_tokens": 256,
  "seed": 42
}
```

Also verify dataset paths, RAG settings, question limit, and output
path. Use a fresh output path if the target CSV already exists; the
experiment runner may refuse to overwrite existing output files.

## Running the Project

### Run a small RAG test

Before running all 100 questions, set the question limit in
`experiment_config.json` to a small value (for example, `2`) and select
a new output path. Make sure Ollama is running, then execute:

``` bash
python run_experiment.py
```

Confirm that the run completes and writes its output.

### Run the full RAG evaluation

After the small test succeeds:

1.  Change the question limit to `100`.
2.  Set a new output path.
3.  Verify the LLM model and RAG configuration.
4.  Run:

``` bash
python run_experiment.py
```

A full RAG evaluation can take considerable time because it runs the
pipeline across the configured questions.

### Run the repeated-run experiment

The repository includes:

``` bash
python run_repeated_experiment.py
```

Check its configuration and output paths before execution. The
experiment is designed for 5 configurations × 5 repeats × 100 questions.

### Surrogate-model experiments

The surrogate implementation is organized under:

``` text
src/surrogate/
```

Important modules: - `preprocessing.py` - `models.py` -
`evaluation.py` - `experiments.py`

The final sample-efficiency results are stored in the output location
listed below. Use the experiment runner/configuration supplied in the
repository to reproduce these experiments. Inspect the script's
arguments and configuration before launching it. If the RAG dataset
already exists, do not rerun the expensive RAG data-generation stage
unnecessarily.

## Important Files

  ------------------------------------------------------------------------------------------------------------------
  File                                                                           Description
  ------------------------------------------------------------------------------ -----------------------------------
  `data/configurations/selected_balanced_v2/train_configs_320.json`              320 training configurations

  `data/configurations/selected_balanced_v2/test_configs_80.json`                Fixed 80 held-out test
                                                                                 configurations

  `data/configurations/search_candidate_pool_1000.json`                          1,000 candidate configurations for
                                                                                 selection experiments

  `data/splits/hotpotqa_100_questions.json`                                      100 HotpotQA questions used

  `data/processed/hotpotqa_100_corpus.json`                                      Processed corpus (997 documents)

  `data/processed/experiments/hotpotqa_config_results.csv`                       Configuration-level aggregate
                                                                                 results (400 rows)

  `data/processed/experiments/hotpotqa_question_results_final_v2.csv`            Question-level results (40,000
                                                                                 rows: 400 × 100)

  `data/processed/experiments/surrogate_results/sample_efficiency_results.csv`   Sample-efficiency experiment
                                                                                 results

  `repeated_question_results.csv`                                                Question-level repeated-run records

  `repeated_run_level_metrics.csv`                                               Metrics for each
                                                                                 configuration-repeat run

  `repeated_run_summary.csv`                                                     Summary across repeated runs
  ------------------------------------------------------------------------------------------------------------------

### Generated plots

The analysis includes plots such as: - `01_rmse_vs_budget.png` -
`02_mae_vs_budget.png` - `03_spearman_vs_budget.png` -
`04_topk_overlap_vs_budget.png` - `05_regret_vs_budget.png` -
`06_quality_vs_latency.png` - `07_quality_vs_token_usage.png` -
Predicted-versus-actual plots for individual models and combined
comparisons.

The exact plot output directory is determined by the experiment
scripts/configuration.

## Reproducibility

For meaningful reproduction: - Use the same question split and corpus. -
Use the same training and held-out configuration files. - Keep the
held-out 80 configurations fixed across budgets. - Preserve the
experiment random seed. - Use identical feature preprocessing and
categorical encoding for training and prediction. - Keep the LLM,
prompt, retrieval pipeline, and evaluation implementation consistent. -
Use fresh output paths or back up existing outputs. - Record hardware,
Python/library versions, Ollama version, and LLM model.

**LLM consistency:** the later local setup uses `qwen2.5:3b` through
Ollama. Historical records should only be compared directly when their
model/provider and pipeline settings match. Changing the LLM can change
answer quality, latency, and token usage.

## Limitations

-   The full configuration space contains 12,000 valid configurations,
    while 400 were evaluated with the actual RAG pipeline.
-   Surrogate predictions are estimates and may be less reliable in
    regions poorly represented by training data.
-   The best model can differ by prediction target.
-   Configuration-selection results are based on a finite candidate pool
    and selected evaluations; they do not guarantee a global optimum.
-   Actual latency may vary with hardware load, model warm-up, and local
    inference conditions.
-   Repeated-run measurements cover selected configurations and should
    not be assumed to represent the variance of every configuration.

## Future Work

-   Evaluate more configurations from the remaining configuration space.
-   Extend experiments to additional datasets.
-   Explore additional surrogate architectures and uncertainty-aware
    selection.
-   Repeat configuration-selection experiments with multiple random
    seeds.
-   Investigate hardware-aware latency prediction.
-   Build an interactive interface where users enter a RAG
    configuration, receive predictions from all five surrogate models,
    run the actual pipeline, and compare predicted versus measured
    metrics.

## Authors

**Project:** Sample-Efficient Surrogate Modeling for RAG Configuration
Performance Prediction

**Repository:**
https://github.com/Kinshukk098/rag-surrogate-optimization

## Acknowledgement

We thank our project supervisor for guidance and feedback throughout the
project.
