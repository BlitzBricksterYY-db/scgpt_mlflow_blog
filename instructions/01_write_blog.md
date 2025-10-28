Given the python notebook, I want to write a technical blog following databricks blog format about how to use Mlflow to log custom pyfunc PythonModel into the Unity Catalog and serve the model on Databricks Model Serving Endpoint with best practice (the do's and dont's).

I am using the example of logging ScGPT, a Foundation Model for Single-Cell Multi-omics Using Generative AI. For this blog, I want to shine on these spotlights at least:
1. how to write the TransformerModelWrapper class as a mlflow.pyfunc.PythonModel class.
  a. how to logging artifacts into the context and load context in .predict() class instance method together with the outside mlflow.pyfunc.log_model(artifacts=) function to specify the artifacts’ local path.
  b. how to pass parameters to class instance method .predict(params=) as new mlflow syntax supports this, and the type hint restrictions of params. It supports flat dictionary with string key and value of scalar or 1-D array of mlflow.types.DataType
  c.  since model serving can only see instance method .predict, vs. unwrap_python_model() can see custom model functions like here the instance method .preprocess(), we need to call .preprocess() inside of .predict(). The parameters are feeded into the .predict() in a way to enable .preprocess() or disable it.
 d. in the ScGPT model example, the .predict(model_input=)  argument receives a pd.DataFrame obj, let us call it ‘input_data'. However, due to the complexity of this example, ‘input_data' is a actually pd.DataFrame container with 1 row and 3 columns, i.e., it contains 3 elements. Each element contains an object needed later to together assembly back into the AnnData object, which can be passed to ScGPT Preprocessor() class for preprocessing. 
	i. if we don’t consider the model serving purpose of the logged model, e.g., we can load the model back from UC in a notebook and do prediction, or even unwrap the model, then we can just define ‘input_data’ with version 1 original format in cell 19. Each element’s data type is very straightforward without further conversion, e.g., ‘adata_sparsematrix’ is a scipy sparse matrix, ‘adata_obs’ is a pd.DataFrame, ‘adata_var’ is also a pd.DataFrame. None of these are serializable.
	ii. however, one of the reason we log the model is we want to model serving it, thus we have to make sure not only the ‘input_data’ but also each element inside 
Is serializable. Thus, ‘input_data’ has version 2 in cell 19. You can see the ‘adata_sparsematrix’ converts to a list of numerical values, ‘adata_obs’ converts to a json string, and ‘adata_var’ also converts to a json string. 
Worth noting, for `pd.DataFrame` class, we could use `.to_json(orient='split’)` to serialize; for `scipy.sparse.csr_matrix`, it needs to first converts to dense array by `.toarray()`, however, `np.ndarray` is still not serializable, so we need to further ’tolist()’ to make it really serializable.
To avoid dimension misalign issue or the broadcasting behaviour of pd.DataFrame, alll elements' values are passed inside the list declaration [] notation, i.e, the square brackets. See the print out
 of ‘input_data’ in cell 20. In this way, both elements and the wrapper pd.DataFrame are serializable.
 	iii. infer_signature() step later in the notebook will correctly infer the data type of each element; signature is needed as an input to model logging; signature will enforce check data type when model is loaded and receive inputs, it will throw error if ‘input_data’ is not of expected format or each element inside is not the expected format.
	iv. During model serving phase, we serialize the serializable ‘input_data’ in cell 20 to string json object. After serialization, it looks like output in cell 38 and 39. Then the string json object will be send via REST api 
POST method to the serving endpoint. 
        v. After the serving endpoint received the serialized ‘input_data’ as string json object, it will convert the string json object back to the original format, i.e., the wrapper will be automatically converted 
back to the pd.DataFrame(). However, for each element inside the wrapper pd.DataFrame, it is untouched. It is your obligation to convert them back to the desired original format ready for next step analysis. Therefore, you need 
to insert some conversion code in the .predict() or other helper functions called by .predict() to convert them back. In this example, in the class .preprocess() methods, we laid down such conversion codes in cell 15 line 91:93.
  
how to log the model using 

Please include everything in the html in the storyline, but with a focus on elaborating the spotlights. You can use both the uploaded html and external web resources to explain the purpose of every code cell. Please include all code cells in the storyline.



——supplemental——
if sparse matrix converts to array instead of to array to list, then in log model step, cannot provide the input_example.
Got error: Invalid input. Data is not compatible with model signature. Failed to convert column adata_sparsematrix to type 'Array(Array(float))'. Error: 'MlflowException('Failed to enforce schema of data `0.0` with dtype `float`')'


