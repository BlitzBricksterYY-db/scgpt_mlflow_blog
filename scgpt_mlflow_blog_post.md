---
title: "Deploying ScGPT Foundation Models with MLflow: A Complete Guide to Custom PyFunc Models and Model Serving"
meta_description: "Learn how to deploy ScGPT foundation models using MLflow custom PyFunc framework. Complete guide covering serialization, model serving, and production deployment best practices."
keywords: "MLflow, ScGPT, foundation models, model serving, PyFunc, Databricks, model deployment, machine learning"
author: "Your Name"
date: "2025-01-07"
tags: ["MLflow", "Foundation Models", "Model Serving", "ScGPT", "Databricks", "Machine Learning"]
---

# Deploying ScGPT Foundation Models with MLflow: A Complete Guide to Custom PyFunc Models and Model Serving

## Introduction

Are you struggling to deploy complex foundation models like ScGPT to production? You're not alone. While MLflow makes model deployment straightforward for traditional ML models, foundation models present unique challenges: complex preprocessing pipelines, custom artifacts, and intricate data serialization requirements. 

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

When you deploy a model using `mlflow.pyfunc.load_model()` in a notebook, you can access custom methods like `preprocess()` using `unwrap_python_model()`. However, **model serving endpoints can only access the `predict` method**. This constraint requires us to:

1. **Embed preprocessing logic** within the `predict` method
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
```python
# Original format - not serializable
input_data = pd.DataFrame({
    'adata_sparsematrix': [adata_subset.X],  # scipy.sparse.csr_matrix
    'adata_obs': [adata_subset.obs],         # pd.DataFrame
    'adata_var': [adata_subset.var]          # pd.DataFrame
})
```

**Version 2 (Model Serving):**
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

# Reconstruct DataFrame from JSON
reconstructed_df = pd.read_json(df_json, orient='split')
```

**Container Strategy:**
```python
# Wrap all elements in lists to avoid broadcasting issues
input_data = pd.DataFrame({
    'column1': [serialized_value1],  # Note the square brackets
    'column2': [serialized_value2],  # This prevents dimension misalignment
    'column3': [serialized_value3]
})
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
adata_subset = adata[:100, :100]  # Use subset for faster logging
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

**3. Signature Inference**: MLflow automatically infers the signature from your input example and parameters.

**4. Artifact Paths**: Use string paths for artifacts - MLflow handles the copying and management.

## Testing and Validation

Before deploying to production, thoroughly test your model:

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

Once your model is logged and tested, you can deploy it to a Databricks Model Serving endpoint:

```python
# Create serving endpoint (via Databricks UI or API)
# The endpoint will automatically handle:
# - Model loading and initialization
# - Request/response serialization
# - Scaling and load balancing
# - Monitoring and logging

# Make inference requests
import requests
import json

serving_input = mlflow.models.convert_input_example_to_serving_input(
    (input_data, params)
)

response = requests.post(
    f"https://{databricks_host}/serving-endpoints/{endpoint_name}/invocations",
    headers={"Authorization": f"Bearer {YOUR_DATABRICKS_TOKEN}"},
    data=serving_input
)

predictions = response.json()
```

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

Ready to deploy your own foundation models? Here are some recommended next steps:

- **Explore the Complete Notebook**: Check out our [full implementation notebook](https://docs.databricks.com/en/mlflow/models.html) with working code
- **Try Different Models**: Apply these patterns to other foundation models like [Geneformer](https://huggingface.co/ctheodoris/Geneformer) or [scBERT](https://www.nature.com/articles/s42256-022-00534-z)
- **Advanced Serving**: Learn about [advanced model serving patterns](https://docs.databricks.com/en/machine-learning/model-serving/index.html) for multi-model endpoints
- **Monitoring & Observability**: Set up [comprehensive monitoring](https://docs.databricks.com/en/machine-learning/model-serving/monitor-model-serving.html) for your deployed models
- **MLflow Documentation**: Deep dive into [MLflow PyFunc models](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and [model signatures](https://mlflow.org/docs/latest/models.html#model-signature-and-input-example)

Have questions or want to share your own deployment experiences? Join the discussion in our [Databricks Community](https://community.databricks.com/) or explore more [MLflow tutorials](https://mlflow.org/docs/latest/tutorials-and-examples/index.html).

---

*This blog post demonstrates advanced MLflow deployment patterns using ScGPT as an example. The techniques shown here apply to many other complex foundation models and can be adapted for your specific use cases.*
