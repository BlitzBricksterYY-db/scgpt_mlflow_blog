---
title: "Deploying ScGPT Foundation Models with MLflow: A Comprehensive Guide to Custom PyFunc Models and Model Serving"
meta_description: "Learn how to deploy ScGPT foundation models using MLflow custom PyFunc framework. Complete guide covering serialization, model serving, and production deployment best practices."
keywords: "MLflow, ScGPT, foundation models, model serving, PyFunc, Databricks, model deployment, machine learning"
author: "Yang Yang, May Merkle-Tan"
date: "2025-10-28"
tags: ["MLflow", "Foundation Models", "Model Serving", "ScGPT", "Databricks", "Machine Learning"]
---

# Deploying ScGPT Foundation Models with MLflow: A Complete Guide to Custom PyFunc Models and Model Serving

## Introduction

Are you struggling to deploy complex models to production? You're not alone. While MLflow makes model deployment straightforward for traditional ML models, foundation models present unique challenges: complex preprocessing pipelines, custom artifacts, and intricate data serialization requirements. 

In this comprehensive guide, we'll walk through deploying ScGPT—a Foundation Model for Single-Cell Multi-omics Using Generative AI—using MLflow's custom PyFunc framework. You'll learn how to overcome the most common deployment hurdles and implement best practices that ensure your models work seamlessly from development to production serving.

## Table of Contents

1. [What You'll Learn](#what-youll-learn)
2. [Understanding ScGPT and the Challenge](#understanding-scgpt-and-the-challenge)
3. [Setting Up the Environment](#setting-up-the-environment)
4. [Creating the Custom PyFunc Model](#creating-the-custom-pyfunc-model)
5. [Implementing the Preprocessing Pipeline](#implementing-the-preprocessing-pipeline)
6. [Implementing the Predict Method](#implementing-the-predict-method)
7. [Solving the Serialization Challenge](#solving-the-serialization-challenge)
8. [Logging the Model with MLflow](#logging-the-model-with-mlflow)
9. [Testing and Validation](#testing-and-validation)
10. [Deploying to Model Serving](#deploying-to-model-serving)
11. [Common Pitfalls and Solutions](#common-pitfalls-and-solutions)
12. [Best Practices Summary](#best-practices-summary)

## What You'll Learn

By the end of this tutorial, you'll master:

- **Custom PyFunc Implementation**: How to wrap complex models using `mlflow.pyfunc.PythonModel`
- **Artifact Management**: Best practices for logging and loading model artifacts
- **Parameter Handling**: Working with MLflow's new parameter syntax and type restrictions
- **Data Serialization**: Solving complex serialization challenges for model serving
- **Production Deployment**: Deploying to Databricks Model Serving endpoints

## Understanding ScGPT and the Challenge

ScGPT is a transformer-based foundation model designed for single-cell genomics analysis. Unlike traditional ML models that work with simple tabular data, ScGPT requires:

1. **Complex preprocessing pipelines** for AnnData objects
2. **Multiple artifacts** (model weights, vocabulary, configuration files)
3. **Specialized data formats** that aren't natively serializable
4. **Custom inference logic** that combines preprocessing and prediction

These requirements make ScGPT an excellent example for learning advanced MLflow deployment patterns.

### Workflow Overview

This section presents two key workflows: first, how to log your ScGPT model to MLflow, and second, how that logged model is deployed and used for inference.

#### ScGPT MLflow Logging Workflow

The following diagram shows the steps to log a ScGPT model to MLflow and Unity Catalog:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ScGPT MLflow Logging Workflow                     │
└─────────────────────────────────────────────────────────────────────────┘

1. PREPARE ARTIFACTS
   ┌───────────────────────────────┐
   │ Gather Model Artifacts:       │
   │ • Model weights (.pt file)    │
   │ • Vocabulary file             │
   │ • Configuration file (JSON)   │
   └──────────┬────────────────────┘
              │
              ▼
   ┌───────────────────────────────┐
   │ Store artifact file paths     │
   └──────────┬────────────────────┘
              │
              ▼

2. CREATE PYFUNC WRAPPER
   ┌──────────────────────────────────┐
   │ Instantiate                      │
   │ TransformerModelWrapper          │
   │ (inherits mlflow.pyfunc.Model)   │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ Define load_context() method     │
   │ - Load artifacts                 │
   │ - Initialize vocab               │
   │ - Load model config              │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ Define predict() method          │
   │ - Preprocessing logic            │
   │ - Model initialization           │
   │ - Inference logic                │
   └──────────┬───────────────────────┘
              │
              ▼

3. PREPARE INPUT EXAMPLE
   ┌──────────────────────────────────┐
   │ Create subset of AnnData         │
   │ adata_subset = adata[:100, :100] │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ Serialize data to DataFrame:     │
   │ • sparse matrix → list of lists  │
   │ • obs DataFrame → JSON string    │
   │ • var DataFrame → JSON string    │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ Wrap in list containers          │
   │ input_data = pd.DataFrame({...}) │
   └──────────┬───────────────────────┘
              │
              ▼

4. DEFINE PARAMETERS
   ┌──────────────────────────────────┐
   │ Create params dictionary:        │
   │ • need_preprocess: True          │
   │ • data_is_raw: False             │
   │ • Model hyperparameters          │
   │ • Preprocessing settings         │
   └──────────┬───────────────────────┘
              │
              ▼

5. INFER SIGNATURE
   ┌──────────────────────────────────┐
   │ mlflow.models.infer_signature()  │
   │ • model_input: input_data        │
   │ • model_output: example output   │
   │ • params: parameters dict        │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ Signature captures:              │
   │ • Input schema                   │
   │ • Output schema                  │
   │ • Parameter types                │
   └──────────┬───────────────────────┘
              │
              ▼

6. SET UP UNITY CATALOG
   ┌──────────────────────────────────┐
   │ mlflow.set_registry_uri(         │
   │   "databricks-uc"                │
   │ )                                │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ Define model name:               │
   │ catalog.schema.model_name        │
   └──────────┬───────────────────────┘
              │
              ▼

7. LOG MODEL
   ┌──────────────────────────────────┐
   │ mlflow.start_run()               │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ mlflow.pyfunc.log_model()        │
   │ • python_model: wrapper          │
   │ • artifacts: file paths dict     │
   │ • signature: inferred schema     │
   │ • input_example: (data, params)  │
   │ • conda_env: dependencies        │
   │ • registered_model_name          │
   └──────────┬───────────────────────┘
              │
              ▼

8. MODEL REGISTERED
   ┌──────────────────────────────────┐
   │ Model logged to MLflow           │
   │ tracking server                  │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ Model registered in              │
   │ Unity Catalog                    │
   │ (catalog.schema.model_name)      │
   └──────────┬───────────────────────┘
              │
              ▼
   ┌──────────────────────────────────┐
   │ ✓ Model ready for deployment     │
   │ ✓ Versioned and tracked          │
   │ ✓ Available for serving          │
   └──────────────────────────────────┘
```

**Key Logging Steps:**

- **Artifact Management**: All model files are referenced by path and MLflow copies them
- **Serialization**: Input examples must be JSON-serializable for model serving
- **Signature**: Defines the contract between model input/output and parameters
- **Unity Catalog**: Centralized model registry with governance and lineage tracking
- **Dependencies**: Conda environment ensures reproducible deployments

#### ScGPT MLflow Deployment Workflow

The following diagram illustrates how a logged ScGPT model is deployed and used for inference:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        ScGPT MLflow Deployment Workflow                  │
└─────────────────────────────────────────────────────────────────────────┘

1. INPUT STAGE
   ┌─────────────────────┐
   │  Raw AnnData Input │
   └──────────┬──────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ Development or Production?   │
   └──────────┬───────────────────┘
              │
      ┌───────┴───────┐
      │               │
   [Development]  [Production]
      │               │
      ▼               ▼
┌──────────────┐  ┌─────────────────────┐
│Direct Format │  │ Serializable Format │
│csr_matrix,   │  │ JSON strings, Lists │
│DataFrame     │  │                     │
└──────┬───────┘  └──────────┬──────────┘
       │                     │
       └──────────┬──────────┘
                  │
                  ▼

2. MODEL INITIALIZATION STAGE
   ┌──────────────────────────────┐
   │ TransformerModelWrapper      │
   │ (PyFunc Model)               │
   └──────────┬───────────────────┘
              │
              ▼
   ┌──────────────────────────────┐
   │ Load Artifacts:              │
   │ • Model weights              │
   │ • Vocabulary                 │
   │ • Configuration              │
   └──────────┬───────────────────┘
              │
              ▼
   ┌─────────────────┐
   │ Initialize Model│
   └──────────┬──────┘
              │
              ▼

3. INFERENCE STAGE
   ┌───────────────┐
   │ Run Inference │
   └───────┬───────┘
           │
           ▼
   ┌──────────────────────┐
   │ All Gene Embeddings │
   │ (30k genes)         │
   └───────┬─────────────┘
           │
           ▼
   ┌──────────────────────┐
   │ Preprocessing Pipeline│
   │ (filter, normalize,   │
   │  binning, HVG)       │
   └──────────┬───────────┘
              │
              ▼
   ┌─────────────────┐
   │  AnnData Object │
   └──────────┬──────┘
              │
              ▼
   ┌──────────────────────┐
   │ Filter by Input Genes?│
   └───────┬──────────────┘
           │
      ┌────┴────┐
      │         │
    [Yes]     [No]
      │         │
      ▼         ▼
┌──────────┐ ┌─────────────────────┐
│Filtered  │ │ All 30k Gene        │
│Embeddings│ │ Embeddings          │
└────┬─────┘ └──────────┬──────────┘
     │                  │
     └────────┬─────────┘
              │
              ▼
   ┌─────────────────┐
   │  Final Output   │
   └────────┬────────┘
            │
            ▼

4. DEPLOYMENT STAGE
   ┌──────────────────────┐
   │   Deployment Type?   │
   └───────┬──────────────┘
           │
      ┌────┴────┐
      │         │
  [Notebook] [Model Serving]
      │         │
      ▼         ▼
┌──────────┐ ┌──────────────────┐
│Local     │ │ HTTP Endpoint    │
│Prediction│ │ REST API         │
└──────────┘ └──────────────────┘
```

**Key Transformation Points:**

- **Input Format**: Raw AnnData objects must be serialized for model serving
- **Preprocessing**: Integrated into the `predict` method for serving compatibility  
- **Model Loading**: Artifacts are loaded through MLflow's context system
- **Output Filtering**: Optionally filters embeddings based on input genes
- **Deployment**: Supports both notebook-based and production serving endpoints

## Setting Up the Environment

First, let's install the required dependencies:

```python
# Install ScGPT and dependencies
pip install scgpt==0.2.4
pip install flash-attn==2.5.9.post1
pip install wandb==0.19.11

# MLflow is typically pre-installed on Databricks
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
```

## Creating the Custom PyFunc Model

The heart of our implementation is the `TransformerModelWrapper` class that extends `mlflow.pyfunc.PythonModel`. This wrapper handles the complexity of ScGPT while providing a clean interface for MLflow.

### The TransformerModelWrapper Class

```python
import json
import pandas as pd
import numpy as np
from typing import Dict
import mlflow.pyfunc
import mlflow.types

class TransformerModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, special_tokens=["<pad>", "<cls>", "<eoc>"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.special_tokens = special_tokens

    def load_context(self, context):
        """Load model artifacts and initialize the model"""
        # Load artifact paths from context
        self.model_file = context.artifacts["model_file"]
        self.model_config_file = context.artifacts["model_config_file"]
        self.vocab_file = context.artifacts["vocab_file"]

        # Initialize vocabulary
        self.vocab = GeneVocab.from_file(self.vocab_file)
        for s in self.special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.gene2idx = self.vocab.get_stoi()
        self.ntokens = len(self.vocab)

        # Load model configuration
        with open(self.model_config_file, "r") as f:
            self.model_configs = json.load(f)
        
        # Extract model parameters
        self.embsize = self.model_configs["embsize"]
        self.nhead = self.model_configs["nheads"]
        self.d_hid = self.model_configs["d_hid"]
        self.nlayers = self.model_configs["nlayers"]
        self.n_layers_cls = self.model_configs["n_layers_cls"]
        self.pad_value = self.model_configs["pad_value"]
        self.mask_value = self.model_configs["mask_value"]
        self.n_bins = self.model_configs["n_bins"]
        self.n_hvg = self.model_configs["n_hvg"]
```

### Key Design Decisions

**1. Artifact Management**: The `load_context` method receives a context object containing paths to all logged artifacts. This approach ensures that model files, configuration, and vocabulary are properly loaded regardless of the deployment environment.

**2. Lazy Loading**: Model artifacts are loaded in `load_context` rather than `__init__`, following MLflow best practices for serialization and deployment.

**3. Device Handling**: The wrapper automatically detects CUDA availability, making it portable across different deployment environments.

## Implementing the Preprocessing Pipeline

One of the most critical aspects of deploying ScGPT is handling the preprocessing pipeline. Since model serving endpoints can only access the `predict` method, we need to integrate preprocessing logic directly into our prediction workflow.

```python
import sys
import os
import json
sys.path.insert(0, "../")
import scgpt
import scanpy
from scgpt.tasks import GeneEmbedding
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.model import TransformerModel
from scgpt.preprocess import Preprocessor
from scgpt.utils import set_seed

import mlflow.pyfunc
import torch
import scipy

os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings("ignore")

from typing import TypedDict, Dict, List, Tuple, Any


class TransformerModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, special_tokens=["<pad>", "<cls>", "<eoc>"]):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.special_tokens = special_tokens

    def load_context(self, context):
        self.model_file = context.artifacts["model_file"]
        self.model_config_file = context.artifacts["model_config_file"]
        self.vocab_file = context.artifacts["vocab_file"]

        self.vocab = GeneVocab.from_file(self.vocab_file)
        for s in self.special_tokens:
            if s not in self.vocab:
                self.vocab.append_token(s)
        self.gene2idx = self.vocab.get_stoi()
        self.ntokens = len(self.vocab)

        with open(self.model_config_file, "r") as f:
            self.model_configs = json.load(f)
        print(
            f"Resume model from {self.model_file}, the model args will override the "
            f"config {self.model_config_file}."
        )
        self.embsize = self.model_configs["embsize"]
        self.nhead = self.model_configs["nheads"]
        self.d_hid = self.model_configs["d_hid"]
        self.nlayers = self.model_configs["nlayers"]
        self.n_layers_cls = self.model_configs["n_layers_cls"]
        self.pad_value = self.model_configs["pad_value"]
        self.mask_value = self.model_configs["mask_value"]
        self.n_bins = self.model_configs["n_bins"]
        self.n_hvg = self.model_configs["n_hvg"]

    def preprocess(
        self,
        context,
        input_data_path: str = None,
        input_dataframe: pd.DataFrame = None,
        data_is_raw=False,
        params={
            "data_is_raw": False,
            "use_key": "X",
            "filter_gene_by_counts": 3,
            "filter_cell_by_counts": False,
            "normalize_total": 1e4,
            "result_normed_key": "X_normed",
            "log1p": False,
            "result_log1p_key": "X_log1p",
            "subset_hvg": 1200,
            "hvg_flavor": "cell_ranger",
            "binning": 51,
            "result_binned_key": "X_binned",
        },
    ):
        if input_data_path and input_dataframe is None:
            loaded_data = scanpy.read(str(input_data_path), cache=True)
            ori_batch_col = "batch"
            loaded_data.obs["celltype"] = loaded_data.obs["final_annotation"].astype(
                str
            )
        elif input_dataframe is not None and isinstance(input_dataframe, pd.DataFrame):
            print(input_dataframe)
            print(input_dataframe.shape)
            #
            adata_sparsematrix = scipy.sparse.csr_matrix(input_dataframe['adata_sparsematrix'][0])
            adata_obs = pd.read_json(input_dataframe['adata_obs'][0], orient='split')
            adata_var = pd.read_json(input_dataframe['adata_var'][0], orient='split')
            loaded_data = scanpy.AnnData(adata_sparsematrix, obs=adata_obs, var=adata_var)
            ori_batch_col = "batch"
            loaded_data.obs["celltype"] = loaded_data.obs["final_annotation"].astype(
                str
            )
        else:
            raise ValueError("No input data provided.")

        self.data_is_raw = params.get("data_is_raw", data_is_raw)

        preprocessor = Preprocessor(
            use_key=params.get("use_key", "X"),
            filter_gene_by_counts=params.get("filter_gene_by_counts", 3),
            filter_cell_by_counts=params.get("filter_cell_by_counts", False),
            normalize_total=params.get("normalize_total", 1e4),
            result_normed_key=params.get("result_normed_key", "X_normed"),
            log1p=params.get("log1p", self.data_is_raw),
            result_log1p_key=params.get("result_log1p_key", "X_log1p"),
            subset_hvg=params.get("subset_hvg", self.n_hvg),
            hvg_flavor=params.get("hvg_flavor", "seurat_v3" if self.data_is_raw else "cell_ranger"),
            binning=params.get("binning", self.n_bins),
            result_binned_key=params.get("result_binned_key", "X_binned"),
        )

        self.n_input_bins = params.get("binning", self.n_bins)
        preprocessor(loaded_data, batch_key=ori_batch_col)

        return loaded_data

    def filter(self, gene_embeddings: np.ndarray, preprocessed_data: AnnData) -> Dict[str, np.ndarray]:
        gene_embeddings = {
            gene: gene_embeddings[i]
            for i, gene in enumerate(self.gene2idx.keys())
            if gene in preprocessed_data.var.index.tolist()
        }
        print("Retrieved gene embeddings for {} genes.".format(len(gene_embeddings)))

        return gene_embeddings

    def predict(
        self,
        context,
        model_input: pd.DataFrame = None,
        params: Dict[str, mlflow.types.DataType] = {
            "need_preprocess": True,
            "input_data_path": None,
            "data_is_raw": False,
            "use_key": "X",
            "filter_gene_by_counts": 3,
            "filter_cell_by_counts": False,
            "normalize_total": 1e4,
            "result_normed_key": "X_normed",
            "log1p": False,
            "result_log1p_key": "X_log1p",
            "subset_hvg": 1200,
            "hvg_flavor": "cell_ranger",
            "binning": 51,
            "result_binned_key": "X_binned",
            "embsize": 512,
            "nhead": 8,
            "d_hid": 512,
            "nlayers": 12,
            "n_layers_cls": 3
        },
    ) -> Dict[str, np.ndarray]:
        print(
            "`model_input` is only needed when users have their own .h5ad gene file to be preprocessed and used to filter the 30k gene embeding result from the model."
        )
        if params["need_preprocess"]:
            assert (
                model_input is not None
            ), "'model_input' must be provided if 'need_preprocess' is True"
            preprocessed_data = self.preprocess(
                context,
                input_data_path=params["input_data_path"],
                input_dataframe=model_input,
                data_is_raw=params["data_is_raw"],
                params=params
            )
            print("preprocessing finished!")

        print("Now defining the TransformerModel!")
        self.model = TransformerModel(
            ntoken=params.get("ntokens", self.ntokens),
            d_model=params.get("embsize", self.embsize),
            nhead=params.get("nhead", self.nhead),
            d_hid=params.get("d_hid", self.d_hid),
            nlayers=params.get("nlayers", self.nlayers),
            vocab=self.vocab,
            pad_value=params.get("pad_value", self.pad_value),
            n_input_bins=params.get(
                "n_input_bins",
                self.n_input_bins if hasattr(self, "n_input_bins") else self.n_bins,
            ),
        )

        try:
            self.model.load_state_dict(torch.load(self.model_file))
            print(f"Loading all model params from {self.model_file}")
        except:
            model_dict = self.model.state_dict()
            pretrained_dict = torch.load(self.model_file)
            pretrained_dict = {
                k: v
                for k, v in pretrained_dict.items()
                if k in model_dict and v.shape == model_dict[k].shape
            }
            for k, v in pretrained_dict.items():
                print(f"Loading params {k} with shape {v.shape}")
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict)

        self.model.to(self.device)
        self.model.eval()

        gene_ids = np.array([id for id in self.gene2idx.values()])
        gene_embeddings = self.model.encoder(
            torch.tensor(gene_ids, dtype=torch.long).to(self.device)
        )
        gene_embeddings = gene_embeddings.detach().cpu().numpy()

        if params["need_preprocess"]:
            gene_embeddings_dict = self.filter(gene_embeddings, preprocessed_data)
        else:
            gene_embeddings_dict = {
                gene: gene_embeddings[i] for i, gene in enumerate(self.gene2idx.keys())
            }
        return gene_embeddings_dict
```

### Why Preprocessing Matters for Model Serving

When you deploy a model using `mlflow.pyfunc.load_model()` in a notebook, you can access custom methods like `.preprocess()` after calling `.unwrap_python_model()` , then handle the preproceessed data to `.predict()` method. However, **model serving endpoints can only access the `.predict()` method**. This constraint requires us to call `.preprocess()` inside of `.predict()` method to handle the preprocessing logic.

1. **Call preprocessing logic** within the `predict` method
2. **Use parameters** to control preprocessing behavior
3. **Handle serialization** of complex data structures

## Implementing the Predict Method

The `predict` method is the entry point for all inference requests. It must handle both preprocessing and prediction while supporting MLflow's parameter system.

```python
def predict(
    self,
    context,
    model_input: pd.DataFrame = None,
    params: Dict[str, mlflow.types.DataType] = {
        "need_preprocess": True,
        "input_data_path": None,
        "data_is_raw": False,
        "use_key": "X",
        "filter_gene_by_counts": 3,
        "filter_cell_by_counts": False,
        "normalize_total": 1e4,
        "result_normed_key": "X_normed",
        "log1p": False,
        "result_log1p_key": "X_log1p",
        "subset_hvg": 1200,
        "hvg_flavor": "cell_ranger",
        "binning": 51,
        "result_binned_key": "X_binned",
        "embsize": 512,
        "nhead": 8,
        "d_hid": 512,
        "nlayers": 12,
        "n_layers_cls": 3
    },
) -> Dict[str, np.ndarray]:
    
    print("`model_input` is only needed when users have their own .h5ad gene file to be preprocessed")
    
    # Conditional preprocessing
    if params["need_preprocess"]:
        assert (
            params["input_data_path"] is not None or model_input is not None
        ), "Either input_data_path or model_input must be provided for preprocessing"
        
        # Call preprocessing
        adata = self.preprocess(
            context,
            input_data_path=params["input_data_path"],
            input_dataframe=model_input,
            data_is_raw=params["data_is_raw"],
            params=params
        )
    else:
        # Skip preprocessing - load data directly
        if params["input_data_path"]:
            adata = sc.read_h5ad(params["input_data_path"])
        else:
            raise ValueError("input_data_path required when preprocessing is disabled")
    
    # Initialize and load the model
    model = TransformerModel(
        ntoken=self.ntokens,
        d_model=params["embsize"],
        nhead=params["nhead"],
        d_hid=params["d_hid"],
        nlayers=params["nlayers"],
        n_layers_cls=params["n_layers_cls"],
        n_cls=1,
        vocab=self.vocab,
        dropout=0.0,
        pad_token=self.special_tokens[0],
        pad_value=self.pad_value,
        adata_gene_col="index",
        adata_gene_col_type="gene_name",
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        num_batch_labels=None,
        domain_spec_batchnorm=False,
        input_emb_style="continuous",
        n_input_bins=self.n_bins,
        cell_emb_style="cls",
        mvc_decoder_style="inner product",
        ecs_threshold=0.0,
        explicit_zero_prob=False,
        use_fast_transformer=True,
        fast_transformer_backend="flash",
    )
    
    # Load model weights
    model.load_state_dict(torch.load(self.model_file, map_location=self.device))
    model.to(self.device)
    model.eval()
    
    # Perform inference
    with torch.no_grad():
        gene_embeddings = model.encode(adata)
    
    # Return results as dictionary
    return {"gene_embeddings": gene_embeddings.cpu().numpy()}
```

### Parameter Handling Best Practices

MLflow's parameter system has specific requirements:

- **Type Restrictions**: Parameters must be scalar values or 1-D arrays of `mlflow.types.DataType`
- **Flat Dictionary**: Nested dictionaries are not supported
- **String Keys**: All parameter keys must be strings
- **Serializable Values**: Values must be JSON-serializable

## Solving the Serialization Challenge

One of the biggest challenges in deploying ScGPT is handling complex data structures that aren't natively serializable. ScGPT works with AnnData objects containing:

- **Sparse matrices** (scipy.sparse.csr_matrix)
- **DataFrames** (pandas.DataFrame)
- **Complex nested structures**

### The Two-Version Approach

We implement two versions of input data formatting:

**Version 1 (Development/Notebook Use):**
This is the original format that is not serializable. It is straightforward and easy to understand. For development and testing purposes in notebook, or deployment without model serving (i.e., just load the model back from UC in a notebook and do prediction), we can directly pass the original format to the .predict() method.
```python
# Original format - not serializable
input_data = pd.DataFrame({
    'adata_sparsematrix': [adata_subset.X],  # scipy.sparse.csr_matrix
    'adata_obs': [adata_subset.obs],         # pd.DataFrame
    'adata_var': [adata_subset.var]          # pd.DataFrame
})
```

**Version 2 (Model Serving):**
Serializable format for model serving. It is more complex. However, it is necessary for model serving. It is the format that is passed to the model serving endpoint.
```python
# Serializable format for model serving
input_data = pd.DataFrame({
    'adata_sparsematrix': [adata_subset.X.toarray().tolist()],  # List of numbers
    'adata_obs': [adata_subset.obs.to_json(orient='split')],    # JSON string
    'adata_var': [adata_subset.var.to_json(orient='split')]     # JSON string
})
```

### Serialization Best Practices

**For Sparse Matrices:**
```python
# Convert sparse matrix to serializable format
sparse_matrix = adata.X  # scipy.sparse.csr_matrix
serializable = sparse_matrix.toarray().tolist()  # List of lists

# Note: .toarray() alone produces np.ndarray which is still not JSON serializable
# Must call .tolist() to get native Python lists
```

**For DataFrames:**
```python
# Convert DataFrame to JSON string
df_json = dataframe.to_json(orient='split')
```

**A DataFrame Container Strategy:**

When sending data to a model serving endpoint, we need to wrap all serialized values in lists (and then wrap each list as a column in a DataFrame container) to ensure proper DataFrame structure being received by the model serving endpoint. Here's why this matters:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    Input Data Structure (DataFrame)                      │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                          DataFrame Container                            │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Column: adata_sparsematrix                                        │ │
│  │ ┌──────────────────────────────────────────────────────────────┐ │ │
│  │ │ [ [ [1,2,3], [4,5,6], ... ] ] ← List container wrapping 2D   │ │ │
│  │ │   ↑ Outer list    ↑ Inner nested lists                        │ │ │
│  │ └──────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Column: adata_obs                                                  │ │
│  │ ┌──────────────────────────────────────────────────────────────┐ │ │
│  │ │ [ "{...json string...}" ] ← List container wrapping JSON     │ │ │
│  │ │   ↑ Outer list    ↑ JSON string                              │ │ │
│  │ └──────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Column: adata_var                                                  │ │
│  │ ┌──────────────────────────────────────────────────────────────┐ │ │
│  │ │ [ "{...json string...}" ] ← List container wrapping JSON     │ │ │
│  │ │   ↑ Outer list    ↑ JSON string                              │ │ │
│  │ └──────────────────────────────────────────────────────────────┘ │ │
│  └────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│                      Why Containers are Needed                           │
└──────────────────────────────────────────────────────────────────────────┘

Model Serving Endpoint
         │
         ▼
┌────────────────────────┐
│ Expects DataFrame      │
│ with single row        │
│ shape = (1, 3)         │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Each column value      │
│ must be wrapped in     │
│ a list container       │
└───────────┬────────────┘
            │
            ▼
┌────────────────────────┐
│ Prevents broadcasting  │
│ and dimension errors   │
└────────────────────────┘
```

**Why We Need A DataFrame Container Strategy:**

Model serving endpoints receive data as a pandas DataFrame with a specific structure. When you pass a DataFrame to the endpoint, each column must contain a single object (or a container holding that object). By wrapping each serialized element in a list, and then wrap each list as a column in a DataFrame container, we can ensure the proper DataFrame structure being received by the model serving endpoint.

1. **Prevents Broadcasting Errors**: Without list containers wrapping the serialized elements, pandas may try to broadcast values across rows, causing dimension mismatches
2. **Ensures Single-Row Structure**: Model serving expects a DataFrame with one row, where each column contains one object
3. **Maintains Data Integrity**: The list containers preserve the exact structure of complex serialized data (nested lists, JSON strings)
4. **Enables Proper Reconstruction**: When the model receives the data, it can reliably extract the first element from each column's list container

```python
# Wrap all elements in lists to avoid broadcasting issues
input_data = pd.DataFrame({
    'adata_sparsematrix': [adata_subset.X.toarray().tolist()],  # List container holding 2D array
    'adata_obs': [adata_subset.obs.to_json(orient='split')],    # List container holding JSON string
    'adata_var': [adata_subset.var.to_json(orient='split')]     # List container holding JSON string
})

# When model receives this DataFrame:
# - input_data.shape = (1, 3)  # Single row, three columns
# - input_data['adata_sparsematrix'][0] = the actual 2D list
# - input_data['adata_obs'][0] = the actual JSON string
```

## Logging the Model with MLflow

Now let's put it all together and log our model to Unity Catalog:

```python
# Set up Unity Catalog
mlflow.set_registry_uri("databricks-uc")
catalog = "your_catalog"
schema = "your_schema"
model_name = "scgpt_gene_embeddings"

# Prepare input example and parameters
params = {
    "need_preprocess": True,
    "input_data_path": None,
    "data_is_raw": False,
    "use_key": "X",
    "filter_gene_by_counts": 3,
    "filter_cell_by_counts": False,
    "normalize_total": 1e4,
    "result_normed_key": "X_normed",
    "log1p": False,
    "result_log1p_key": "X_log1p",
    "subset_hvg": 1200,
    "hvg_flavor": "cell_ranger",
    "binning": 51,
    "result_binned_key": "X_binned",
    "embsize": 512,
    "nhead": 8,
    "d_hid": 512,
    "nlayers": 12,
    "n_layers_cls": 3
}

# Create serializable input example
adata_subset = adata[:100, :100]  # Use subset for example purpose
input_data = pd.DataFrame({
    'adata_sparsematrix': [adata_subset.X.toarray().tolist()],
    'adata_obs': [adata_subset.obs.to_json(orient='split')],
    'adata_var': [adata_subset.var.to_json(orient='split')]
})

# Infer signature
signature = infer_signature(
    model_input=input_data,
    model_output={"gene_embeddings": np.random.rand(100, 512)},  # Example output
    params=params
)

# Log the model
with mlflow.start_run() as run:
    registered_model_name = f"{catalog}.{schema}.{model_name}"
    
    mlflow.pyfunc.log_model(
        "model",
        python_model=TransformerModelWrapper(
            special_tokens=["<pad>", "<cls>", "<eoc>"]
        ),
        artifacts={
            "model_file": str(model_file_path),
            "model_config_file": str(config_file_path),
            "vocab_file": str(vocab_file_path),
        },
        conda_env={
            'name': 'mlflow-env',
            'channels': ['conda-forge'],
            'dependencies': [
                'python=3.11.11',
                'pip',
                {
                    'pip': [
                        'numpy==1.26.4',
                        'torch==2.0.1+cu118',
                        'torchvision==0.15.2+cu118',
                        'scgpt==0.2.4',
                        'flash-attn==2.5.8+cu118',
                        'wandb==0.19.11',
                    ],
                },
            ],
        },
        signature=signature,
        input_example=(input_data, params),  # Tuple format for input + params
        registered_model_name=registered_model_name,
    )
```

### Key Logging Considerations

**1. Conda Environment**: Specify exact versions to ensure reproducibility across environments.

**2. Input Example**: Use tuple format `(input_data, params)` to provide both data and parameters as examples.

**3. Signature Inference**: MLflow automatically infers the signature by calling `infer_signature()` on your input example, output example and parameters. This signature is used to validate the input data, output data and parameters when the model is loaded and used for inference. It will throw error if the input data, output data and parameters are not of the expected format. 
  - An MLflow signature defines the expected inputs and outputs of a model—essentially acting as a contract that ensures data passed into the model during inference matches the format it was trained on. This prevents mismatches, improves reproducibility, and makes deployment safer and more transparent.

**4. Artifact Paths**: Use string paths for artifacts - MLflow handles the copying and management.

## Testing and Validation

Before deploying to production, thoroughly test your model:
1. testing by loading the model and calling the `.predict()` method with the input example and parameters in the notebook. Good for debugging and testing the model as well as deploying a batch job (e.g., a job with notebooks as tasks).

```python
# Load the model
model = mlflow.pyfunc.load_model(f"models:/{registered_model_name}/1")

# Test prediction
predictions = model.predict(
    data=input_data,
    params={
        'need_preprocess': True,
        'input_data_path': None,
        'data_is_raw': False,
        'use_key': 'X',
        # ... other parameters
    }
)
```

2. Before creating the model serving endpoint, testing serialization and validating the serving input by calling `.convert_input_example_to_serving_input()` and `.validate_serving_input()`. This is to ensure the input example and parameters are serialized correctly and can be used by the model serving endpoint.

```python
# Validate serving input
serving_input_example = mlflow.models.convert_input_example_to_serving_input(
    (input_data, params)
)
mlflow.models.validate_serving_input(
    f"models:/{registered_model_name}/1", 
    serving_input_example
)
```

## Deploying to Model Serving

Once your model is logged and tested, you can deploy it to a Databricks Model Serving endpoint. This section covers creating the endpoint and making inference requests.

### Creating Model Serving Endpoint and Deploying

Databricks offers multiple ways to create model serving endpoints. Each method has its own advantages depending on your use case, team expertise, and deployment requirements. Here are all four approaches, ranked by recommendation:

#### Method 1: Databricks SDK for Python (Recommended for Production)

The Databricks SDK for Python is the most modern and recommended approach for production deployments and automation.

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

# Initialize workspace client (automatically uses your credentials)
w = WorkspaceClient()

# Create serving endpoint
endpoint = w.serving_endpoints.create(
    name="scgpt-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog}.{schema}.scgpt_gene_embeddings",
                entity_version="1",
                workload_size="Medium",
                scale_to_zero_enabled=False,
                workload_type="GPU_SMALL"  # GPU recommended for ScGPT
            )
        ]
    )
)

print(f"✓ Endpoint '{endpoint.name}' created successfully")
print(f"  State: {endpoint.state.config_update}")

# Wait for endpoint to be ready
w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint.name)
print(f"✓ Endpoint is ready for serving")
```

**Pros:**
- ✅ **Type-safe**: Full IDE autocomplete and type checking
- ✅ **Official SDK**: Long-term support from Databricks
- ✅ **Automatic authentication**: Uses Databricks CLI or environment credentials
- ✅ **Built-in retry logic**: Handles transient failures automatically
- ✅ **Consistent API**: Works seamlessly with other Databricks services
- ✅ **Better error handling**: Clear, actionable error messages
- ✅ **Wait operations**: Built-in helpers to wait for async operations

**Cons:**
- ❌ **Additional dependency**: Requires `databricks-sdk` package
- ❌ **Learning curve**: Need to learn SDK-specific patterns (minor)

**Best for:** CI/CD pipelines, Infrastructure as Code, production automation, teams managing multiple endpoints

#### Method 2: MLflow Deployments SDK (Recommended for MLflow-Centric Workflows)

If your team is already heavily invested in MLflow, the MLflow Deployments SDK provides a unified interface for model deployment.

```python
from mlflow.deployments import get_deploy_client

# Initialize deployment client for Databricks
client = get_deploy_client("databricks")

# Create endpoint
endpoint = client.create_endpoint(
    name="scgpt-serving",
    config={
        "served_entities": [{
            "entity_name": f"{catalog}.{schema}.scgpt_gene_embeddings",
            "entity_version": "1",
            "workload_size": "Medium",
            "scale_to_zero_enabled": False,
            "workload_type": "GPU_SMALL"
        }]
    }
)

print(f"Endpoint created: {endpoint['name']}")

# Get endpoint details
endpoint_info = client.get_endpoint(endpoint="scgpt-serving")
print(f"Endpoint state: {endpoint_info['state']}")
```

**Pros:**
- ✅ **MLflow integration**: Seamless with existing MLflow workflows
- ✅ **Unified API**: Same interface for different deployment targets
- ✅ **Simpler syntax**: More straightforward for basic use cases
- ✅ **Model-centric**: Focuses on model deployment abstraction
- ✅ **No extra dependencies**: Usually already installed with MLflow

**Cons:**
- ❌ **Less comprehensive**: Fewer features than full Databricks SDK
- ❌ **Limited to model serving**: Doesn't cover other Databricks operations
- ❌ **Less type safety**: More dict-based configuration
- ❌ **Deployment-specific**: Only useful for serving endpoints

**Best for:** MLflow-first organizations, model-centric teams, simpler deployment workflows

#### Method 3: Databricks UI (Recommended for Initial Exploration)

The graphical interface is ideal for learning, experimentation, and manual management.

**Steps:**

1. Navigate to **Models** in the Databricks workspace left sidebar
2. Select your registered model (e.g., `catalog.schema.scgpt_gene_embeddings`)
3. Click on the **Serving** tab at the top
4. Click **Create serving endpoint** button
5. Configure the endpoint settings:
   - **Endpoint name**: Enter a descriptive name (e.g., `scgpt-serving`)
   - **Model version**: Select the version to serve (e.g., Version 1)
   - **Compute size**: Choose workload size (Small, Medium, Large)
   - **Compute type**: Select GPU_SMALL for ScGPT models
   - **Scale to zero**: Disable for production, enable for cost savings
   - **Rate limiting** (optional): Set requests per minute limits
6. Review the configuration summary
7. Click **Create** to deploy the endpoint

**Pros:**
- ✅ **No code required**: Perfect for non-programmers
- ✅ **Visual feedback**: See all options and validations immediately
- ✅ **Easy to explore**: Discover features through UI elements
- ✅ **Quick setup**: Fastest way to get started initially
- ✅ **Built-in help**: Tooltips and descriptions for all options
- ✅ **Real-time monitoring**: Visual dashboard for endpoint metrics

**Cons:**
- ❌ **Not reproducible**: Manual steps are hard to replicate exactly
- ❌ **No version control**: Can't track changes in git
- ❌ **Manual process**: Time-consuming for multiple endpoints
- ❌ **Not automatable**: Can't integrate into CI/CD pipelines
- ❌ **Team collaboration**: Harder to share exact configurations

**Best for:** Learning and prototyping, one-off deployments, exploring capabilities, debugging configurations

#### Method 4: REST API (For Custom Integrations)

Direct HTTP API calls provide maximum flexibility for custom integrations and non-Python environments.

```python
import requests
import json

# Databricks workspace configuration
databricks_host = "your-workspace.cloud.databricks.com"
databricks_token = "YOUR_DATABRICKS_TOKEN"

# Create serving endpoint
url = f"https://{databricks_host}/api/2.0/serving-endpoints"
headers = {
    "Authorization": f"Bearer {databricks_token}",
    "Content-Type": "application/json"
}

endpoint_config = {
    "name": "scgpt-serving",
    "config": {
        "served_entities": [{
            "entity_name": f"{catalog}.{schema}.scgpt_gene_embeddings",
            "entity_version": "1",
            "workload_size": "Medium",
            "scale_to_zero_enabled": False,
            "workload_type": "GPU_SMALL"
        }]
    }
}

response = requests.post(url, json=endpoint_config, headers=headers)

if response.status_code == 200:
    endpoint = response.json()
    print(f"✓ Endpoint created: {endpoint['name']}")
else:
    print(f"✗ Error: {response.status_code}")
    print(f"  Message: {response.json()}")

# Check endpoint status
status_url = f"https://{databricks_host}/api/2.0/serving-endpoints/scgpt-serving"
status_response = requests.get(status_url, headers=headers)
print(f"Endpoint state: {status_response.json()['state']}")
```

**Using curl (for non-Python environments):**

```bash
# Set your Databricks configuration
DATABRICKS_HOST="your-workspace.cloud.databricks.com"
DATABRICKS_TOKEN="YOUR_DATABRICKS_TOKEN"
CATALOG="your_catalog"
SCHEMA="your_schema"

# Create serving endpoint
curl -X POST "https://${DATABRICKS_HOST}/api/2.0/serving-endpoints" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "scgpt-serving",
    "config": {
      "served_entities": [{
        "entity_name": "'"${CATALOG}.${SCHEMA}"'.scgpt_gene_embeddings",
        "entity_version": "1",
        "workload_size": "Medium",
        "scale_to_zero_enabled": false,
        "workload_type": "GPU_SMALL"
      }]
    }
  }'

# Check endpoint status
curl -X GET "https://${DATABRICKS_HOST}/api/2.0/serving-endpoints/scgpt-serving" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json"

# Get endpoint state with pretty print
curl -X GET "https://${DATABRICKS_HOST}/api/2.0/serving-endpoints/scgpt-serving" \
  -H "Authorization: Bearer ${DATABRICKS_TOKEN}" \
  -H "Content-Type: application/json" | jq '.state'
```

**Pros:**
- ✅ **Language agnostic**: Works with any programming language
- ✅ **Maximum flexibility**: Direct control over all API parameters
- ✅ **Minimal dependencies**: Only needs HTTP client library
- ✅ **Custom integrations**: Easy to integrate into existing systems
- ✅ **Direct access**: No abstraction layers

**Cons:**
- ❌ **Verbose**: More boilerplate code required
- ❌ **Manual auth management**: Must handle tokens and credentials
- ❌ **No type safety**: Easy to make mistakes in request payloads
- ❌ **Manual error handling**: Must implement retries and error logic
- ❌ **API version management**: Need to track API changes manually
- ❌ **Less discoverable**: Must reference API documentation constantly

**Best for:** Non-Python environments (Java, Go, etc.), legacy system integration, custom tooling, polyglot teams

---

### Comparison Summary

| Feature | Databricks SDK | MLflow SDK | UI | REST API |
|---------|---------------|------------|-----|----------|
| **Ease of Use** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Production Ready** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ |
| **Automation** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ |
| **Type Safety** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | N/A | ⭐ |
| **Learning Curve** | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Flexibility** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |

### Our Recommendation

For **production deployments and team collaboration**, use the **Databricks SDK** (Method 1). It provides the best balance of ease of use, type safety, and long-term maintainability.

For **quick experiments or learning**, start with the **UI** (Method 3) to understand the options, then translate to code using the Databricks SDK.

For **MLflow-heavy workflows**, the **MLflow Deployments SDK** (Method 2) offers seamless integration with your existing MLflow-based systems.

Use the **REST API** (Method 4) only when you need to integrate with non-Python systems or have specific requirements that other methods don't support.

### Endpoint Features

Regardless of which method you choose, the serving endpoint automatically provides:
- **Model Loading**: Artifacts are loaded from Unity Catalog
- **Request/Response Serialization**: JSON encoding/decoding
- **Scaling**: Auto-scaling based on traffic (or scale-to-zero for cost savings)
- **Load Balancing**: Distributes requests across instances
- **Monitoring**: Built-in metrics, logs, and dashboards
- **Authentication**: Token-based access control
- **Versioning**: Support for A/B testing and blue-green deployments

### Model Inference

Once your serving endpoint is deployed and ready, you can make inference requests via HTTP POST calls.

**Request Format:**

```python
import requests
import json
import pandas as pd

# Prepare serializable input data
adata_subset = adata[:50, :100]  # Use a smaller subset for testing
input_data = pd.DataFrame({
    'adata_sparsematrix': [adata_subset.X.toarray().tolist()],
    'adata_obs': [adata_subset.obs.to_json(orient='split')],
    'adata_var': [adata_subset.var.to_json(orient='split')]
})

# Prepare parameters
params = {
    "need_preprocess": True,
    "input_data_path": None,
    "data_is_raw": False,
    "use_key": "X",
    "filter_gene_by_counts": 3,
    "filter_cell_by_counts": False,
    "normalize_total": 1e4,
    "result_normed_key": "X_normed",
    "log1p": False,
    "result_log1p_key": "X_log1p",
    "subset_hvg": 1200,
    "hvg_flavor": "cell_ranger",
    "binning": 51,
    "result_binned_key": "X_binned",
    "embsize": 512,
    "nhead": 8,
    "d_hid": 512,
    "nlayers": 12,
    "n_layers_cls": 3
}

# Convert to serving format
serving_input = mlflow.models.convert_input_example_to_serving_input(
    (input_data, params)
)

# Make inference request
response = requests.post(
    f"https://{databricks_host}/serving-endpoints/scgpt-serving/invocations",
    headers={"Authorization": f"Bearer {YOUR_DATABRICKS_TOKEN}"},
    data=serving_input
)

# Handle response
if response.status_code == 200:
    predictions = response.json()
    print(f"Embeddings shape: {predictions['gene_embeddings'].shape if isinstance(predictions['gene_embeddings'], list) else 'Check response'}")
    print(f"Successfully retrieved gene embeddings!")
else:
    print(f"Error: {response.status_code}")
    print(f"Response: {response.text}")
```

**Response Format:**

The model serving endpoint returns predictions in JSON format:

```json
{
    "predictions": [
        {
            "gene_embeddings": [[0.123, 0.456, ...], [...], ...]  // Nested list of embeddings
        }
    ]
}
```

**Handling Large Batches:**

For larger datasets, you may want to process in batches:

```python
import pandas as pd
import numpy as np

def batch_inference(model_endpoint, data_list, params, batch_size=10):
    """
    Make batch inference requests to the model serving endpoint
    
    Args:
        model_endpoint: URL of the serving endpoint
        data_list: List of AnnData objects or DataFrame objects
        params: Model parameters
        batch_size: Number of samples per batch
    
    Returns:
        List of predictions
    """
    all_predictions = []
    
    for i in range(0, len(data_list), batch_size):
        batch_data = data_list[i:i+batch_size]
        
        # Prepare batch input
        batch_input_data = pd.DataFrame({
            'adata_sparsematrix': [data.X.toarray().tolist() for data in batch_data],
            'adata_obs': [data.obs.to_json(orient='split') for data in batch_data],
            'adata_var': [data.var.to_json(orient='split') for data in batch_data]
        })
        
        # Convert to serving format
        serving_input = mlflow.models.convert_input_example_to_serving_input(
            (batch_input_data, params)
        )
        
        # Make request
        response = requests.post(
            model_endpoint,
            headers={"Authorization": f"Bearer {YOUR_DATABRICKS_TOKEN}"},
            data=serving_input
        )
        
        if response.status_code == 200:
            batch_predictions = response.json()
            all_predictions.extend(batch_predictions['predictions'])
        else:
            print(f"Error in batch {i//batch_size}: {response.text}")
    
    return all_predictions

# Example usage
embeddings = batch_inference(
    model_endpoint=f"https://{databricks_host}/serving-endpoints/scgpt-serving/invocations",
    data_list=[adata_subset1, adata_subset2, adata_subset3],
    params=params,
    batch_size=5
)
```

### Alternative Methods for Making Inference Requests

Databricks Model Serving provides multiple ways to query your deployed models, each suited for different use cases and workflows:

- **Serving UI**: Interactive web interface in the Databricks workspace where you can manually test your endpoint by inserting JSON input data and sending requests with a click. Ideal for quick testing and debugging during development.

- **SQL Function (`ai_query`)**: Invoke model inference directly from SQL queries using the `ai_query()` function. Perfect for integrating ML predictions into existing data pipelines, dashboards, and analytics workflows without leaving SQL.

- **REST API**: Standard HTTP POST requests to the endpoint's `/invocations` URL. Most flexible option that works with any programming language or tool that can make HTTP requests (curl, Postman, etc.).

- **MLflow Deployments SDK**: Python-based `predict()` function from the MLflow Deployments SDK. Best for Python-centric workflows and provides a cleaner abstraction over raw REST API calls.

- **PowerBI Integration**: Query models directly from PowerBI Desktop using custom M queries. Enables real-time predictions in business intelligence reports and dashboards.

- **Databricks Notebooks**: Direct Python/Scala/R integration within Databricks notebooks for interactive data science and ML workflows.

For detailed examples and code snippets for each method, refer to the [Databricks Model Serving documentation](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/model-serving/score-custom-model-endpoints).

# Monitoring Model Serving

There is a whole section on monitoring Mosaic AI model serving in the [Databricks Model Serving documentation](https://docs.databricks.com/aws/en/machine-learning/model-serving/monitor-diagnose-endpoints). Serving ScGPT falls under the Mosaic AI model serving category. Here we will show the examples of:  
1. **monitoring and debugging the endpoint.**  
2. **monitoring and analyzing the inference data.**


**Monitoring and Debugging of Endpoint:**

```python
# Check endpoint status
import requests

response = requests.get(
    f"https://{databricks_host}/api/2.0/serving-endpoints/scgpt-serving",
    headers={"Authorization": f"Bearer {YOUR_DATABRICKS_TOKEN}"}
)
endpoint_info = response.json()

print(f"Endpoint State: {endpoint_info['state']}")
print(f"Served Models: {endpoint_info['config']['served_models']}")
print(f"Creation Time: {endpoint_info['creation_timestamp']}")

# View recent logs
if 'logs' in endpoint_info:
    print(f"Recent Logs: {endpoint_info['logs']}")
```

## Monitoring Model Serving with Inference Tables

Effective monitoring is critical for maintaining reliable model serving in production. Databricks provides built-in inference tables that automatically log all requests and responses, enabling you to track performance, debug issues, and analyze model behavior over time.

### Inference Tables for Payload Logging

Inference tables automatically capture every request and response sent to your serving endpoint, storing them in Delta tables for analysis and auditing. This is essential for production monitoring, compliance, and debugging.

Databricks now provides **AI Gateway-enabled inference tables**, which offer enhanced monitoring capabilities including automatic cost tracking, usage analytics, and better governance features compared to legacy inference tables. We prefer this way instead of the legacy inference tables `.inference_table_config` in the endpoint configuration.

To use AI Gateway inference tables, you must enable it through Databricks SDK or REST API or UI, **but not through the MLflow SDK**. Here we will show the examples of using **Databricks SDK** to enable AI Gateway-enabled inference tables.

**Prerequisites:**
- Unity Catalog must be enabled in your workspace
- Serverless compute should be enabled for optimal performance
- You need `USE CATALOG`, `USE SCHEMA`, and `CREATE TABLE` permissions
- Workspace must be in a region where model serving is supported

**Benefits of AI Gateway-enabled inference tables:**
- **Automatic Payload Logging**: Captures all requests and responses
- **Cost Tracking**: Built-in usage and cost monitoring per endpoint
- **Governance**: Enhanced compliance and audit trail features
- **Performance**: Optimized for high-throughput workloads
- **Unity Catalog Integration**: Seamless Delta table management

**Enable AI Gateway-enabled inference tables when creating your endpoint:**

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, 
    ServedEntityInput,
    AiGatewayConfig,
    AiGatewayInferenceTableConfig
)

w = WorkspaceClient()

# Create endpoint with AI Gateway-enabled inference table
endpoint = w.serving_endpoints.create(
    name="scgpt-serving",
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=f"{catalog}.{schema}.scgpt_gene_embeddings",
                entity_version="1",
                workload_size="Medium",
                scale_to_zero_enabled=False,
                workload_type="GPU_SMALL"
            )
        ],
        # Enable AI Gateway with inference tables for automatic payload logging
        auto_capture_config=AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=catalog,
                schema_name=schema,
                table_name_prefix="scgpt_inference",
                enabled=True
            ),
            usage_tracking_config={
                "enabled": True  # Enable cost and usage tracking
            }
        )
    )
)

print(f"✓ Endpoint created with AI Gateway-enabled inference table")
print(f"  Inference data will be logged to: {catalog}.{schema}.scgpt_inference_payload")
print(f"  Cost tracking enabled for monitoring usage")
```

**Update existing endpoint to enable AI Gateway-enabled inference tables:**

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    AiGatewayConfig,
    AiGatewayInferenceTableConfig
)

w = WorkspaceClient()

# Update endpoint configuration with AI Gateway
w.serving_endpoints.update_config(
    name="scgpt-serving",
    served_entities=[
        ServedEntityInput(
            entity_name=f"{catalog}.{schema}.scgpt_gene_embeddings",
            entity_version="1",
            workload_size="Medium",
            scale_to_zero_enabled=False,
            workload_type="GPU_SMALL"
        )
    ],
    auto_capture_config=AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=catalog,
            schema_name=schema,
            table_name_prefix="scgpt_inference",
            enabled=True
        ),
        usage_tracking_config={
            "enabled": True
        }
    )
)

print("✓ AI Gateway-enabled inference table enabled for existing endpoint")
```

### Analyzing Inference Data

Once AI Gateway-enabled inference tables are enabled, you can query the logged data for monitoring and analysis. The AI Gateway inference tables include enhanced fields for better monitoring and governance.

**AI Gateway Inference Table Schema:**

AI Gateway-enabled inference tables include the following key columns:
- `databricks_request_id`: Unique identifier assigned by Databricks
- `client_request_id`: Optional client-provided request ID
- `request_date`: Date partition for efficient querying
- `request_time`: Timestamp when request was received
- `status_code`: HTTP status code of the response
- `execution_duration_ms`: Time taken to process the request
- `request`: Input data sent to the model
- `response`: Output data returned by the model
- `served_entity_id`: Identifier of the model version that processed the request
- `sampling_fraction`: Sampling rate if sampling is enabled
- `requester`: Identity of the user or service that made the request
- `logging_error_codes`: Any errors during logging

**View recent inference requests:**

```python
# Query the AI Gateway inference table using Spark SQL
inference_df = spark.sql(f"""
    SELECT 
        databricks_request_id,
        client_request_id,
        request_date,
        request_time,
        status_code,
        execution_duration_ms,
        served_entity_id,
        requester,
        request,
        response
    FROM {catalog}.{schema}.scgpt_inference_payload
    WHERE request_date >= current_date() - INTERVAL 7 DAYS
    ORDER BY request_time DESC
    LIMIT 100
""")

display(inference_df)
```

**Analyze performance metrics:**

```python
# Calculate key performance indicators
performance_metrics = spark.sql(f"""
    SELECT 
        request_date,
        COUNT(*) as total_requests,
        AVG(execution_duration_ms) as avg_latency_ms,
        PERCENTILE(execution_duration_ms, 0.50) as p50_latency_ms,
        PERCENTILE(execution_duration_ms, 0.95) as p95_latency_ms,
        PERCENTILE(execution_duration_ms, 0.99) as p99_latency_ms,
        SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) as error_count,
        SUM(CASE WHEN status_code != 200 THEN 1 ELSE 0 END) / COUNT(*) * 100 as error_rate_pct,
        COUNT(DISTINCT served_entity_id) as model_versions_used,
        COUNT(DISTINCT requester) as unique_requesters
    FROM {catalog}.{schema}.scgpt_inference_payload
    WHERE request_date >= current_date() - INTERVAL 30 DAYS
    GROUP BY request_date
    ORDER BY request_date DESC
""")

display(performance_metrics)
```

**Identify slow requests:**

```python
# Find requests with high latency
slow_requests = spark.sql(f"""
    SELECT 
        databricks_request_id,
        request_time,
        execution_duration_ms,
        status_code,
        served_entity_id,
        requester,
        request
    FROM {catalog}.{schema}.scgpt_inference_payload
    WHERE request_date >= current_date() - INTERVAL 1 DAYS
      AND execution_duration_ms > 5000  -- Requests taking more than 5 seconds
    ORDER BY execution_duration_ms DESC
    LIMIT 20
""")

display(slow_requests)
```

**Monitor error patterns:**

```python
# Analyze errors by status code
error_analysis = spark.sql(f"""
    SELECT 
        status_code,
        COUNT(*) as error_count,
        MIN(request_time) as first_occurrence,
        MAX(request_time) as last_occurrence,
        COUNT(DISTINCT requester) as affected_users,
        COLLECT_LIST(databricks_request_id)[0:5] as sample_request_ids,
        COLLECT_LIST(DISTINCT served_entity_id) as affected_model_versions
    FROM {catalog}.{schema}.scgpt_inference_payload
    WHERE request_date >= current_date() - INTERVAL 7 DAYS
      AND status_code >= 400
    GROUP BY status_code
    ORDER BY error_count DESC
""")

display(error_analysis)
```

### Best Practices for Inference Tables

1. **Use AI Gateway**: Enable AI Gateway-enabled inference tables for new deployments to get enhanced monitoring and cost tracking
2. **Enable Early**: Turn on inference tables during initial endpoint creation
3. **Regular Monitoring**: Query inference data at least daily for production endpoints
4. **Data Retention**: Set up retention policies to manage table size (e.g., keep 90 days)
5. **Alert Thresholds**: Define clear thresholds for latency, error rates, and volume
6. **Performance Analysis**: Use inference data to optimize model performance and costs
7. **Compliance**: Leverage logged data for audit trails and regulatory requirements
8. **Cost Tracking**: Monitor usage metrics enabled by AI Gateway to optimize resource allocation

**Migration Note for Legacy Inference Tables:**

If you have existing endpoints using legacy `InferenceTableConfig`, you can migrate to AI Gateway-enabled inference tables:

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# First, disable legacy inference tables
w.serving_endpoints.update_config(
    name="scgpt-serving",
    served_entities=[...],
    inference_table_config=None  # Disable legacy tables
)

# Then, enable AI Gateway-enabled inference tables
w.serving_endpoints.update_config(
    name="scgpt-serving",
    served_entities=[...],
    auto_capture_config=AiGatewayConfig(
        inference_table_config=AiGatewayInferenceTableConfig(
            catalog_name=catalog,
            schema_name=schema,
            table_name_prefix="scgpt_inference",
            enabled=True
        ),
        usage_tracking_config={"enabled": True}
    )
)
```

For more information on AI Gateway and inference tables, see the [Databricks AI Gateway documentation](https://docs.databricks.com/en/ai-gateway/inference-tables.html).

## Common Pitfalls and Solutions

### 1. Serialization Errors
**Problem**: `TypeError: Object of type csr_matrix is not JSON serializable`

**Solution**: Always convert complex objects to basic Python types:
```python
# Wrong
sparse_matrix = adata.X  # scipy.sparse.csr_matrix

# Right
sparse_list = adata.X.toarray().tolist()  # List of lists
```

### 2. Parameter Type Errors
**Problem**: `MlflowException: Failed to enforce schema of data`

**Solution**: Ensure all parameters are scalar or 1-D arrays of supported types:
```python
# Wrong
params = {"nested": {"key": "value"}}

# Right
params = {"key": "value"}  # Flat dictionary only
```

### 3. Dimension Misalignment
**Problem**: Broadcasting errors when reconstructing DataFrames

**Solution**: Always wrap values in lists:
```python
# Wrong
pd.DataFrame({'col': value})

# Right
pd.DataFrame({'col': [value]})  # Note the brackets
```

### 4. Missing Dependencies
**Problem**: Model fails to load due to missing packages

**Solution**: Specify complete conda environment with exact versions:
```python
conda_env = {
    'dependencies': [
        'python=3.11.11',
        {'pip': ['scgpt==0.2.4', 'flash-attn==2.5.8']}
    ]
}
```

## Best Practices Summary

### Development Best Practices
- **Start Simple**: Begin with a minimal working example before adding complexity
- **Test Locally**: Always test your PyFunc model locally before logging
- **Version Control**: Use clear versioning for your models and track changes
- **Documentation**: Document your preprocessing steps and parameter meanings

### Production Best Practices
- **Environment Consistency**: Use identical package versions across dev/prod
- **Input Validation**: Validate inputs thoroughly in your `predict` method
- **Error Handling**: Implement robust error handling and logging
- **Monitoring**: Set up monitoring for model performance and errors

### Performance Best Practices
- **Lazy Loading**: Load heavy artifacts only when needed
- **Caching**: Cache preprocessed data when possible
- **Batch Processing**: Design your model to handle batch inputs efficiently
- **Resource Management**: Properly manage GPU memory and cleanup resources

## Conclusion

Deploying complex foundation models like ScGPT with MLflow requires careful attention to serialization, parameter handling, and preprocessing integration. By following the patterns outlined in this guide, you can successfully deploy sophisticated models to production while maintaining the flexibility and scalability that MLflow provides.

The key takeaways are:

1. **Custom PyFunc models** provide the flexibility needed for complex foundation models
2. **Proper serialization** is critical for model serving compatibility
3. **Parameter-driven preprocessing** enables flexible inference workflows
4. **Thorough testing** prevents deployment issues

## Next Steps

While this blog is comprehensive, it is not exhaustive. Ready to deploy your own foundation models? Here are some recommended next steps:

- **Explore the Complete Notebook**: Check out our [full implementation notebook](https://github.com/databricks-ia/scgpt-mlflow-blog-post/blob/main/scgpt_mlflow_blog_post.ipynb) with working code
- **Vibe coding leveraging this Blog Post**: The quickest way to deploy your own foundation models is to reference this blog post in your vibe coding tool. Make sure you also learn through the blog post to understand the concepts and patterns.
- **Try Different Models**: Apply these patterns to other foundation models like [Geneformer](https://huggingface.co/ctheodoris/Geneformer) or [scBERT](https://www.nature.com/articles/s42256-022-00534-z)
- **Mosaic AI Model Serving**: Learn about [other Mosaic AI model serving patterns](https://docs.databricks.com/en/machine-learning/model-serving/index.html) for multi-model endpoints
- **Monitoring & Observability**: Set up [comprehensive monitoring](https://docs.databricks.com/aws/en/machine-learning/model-serving/monitor-diagnose-endpoints) for your deployed models
- **MLflow Documentation**: Deep dive into [MLflow PyFunc models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and [model signatures](https://mlflow.org/docs/latest/models.html#model-signature-and-input-example)

Have questions or want to share your own deployment experiences? Join the discussion in our [Databricks Community](https://community.databricks.com/) or explore more [MLflow tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html).

---

*This blog post demonstrates advanced MLflow deployment patterns using ScGPT as an example. The techniques shown here apply to many other complex foundation models and can be adapted for your specific use cases.*
