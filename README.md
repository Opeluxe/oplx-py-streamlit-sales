# oplx-py-streamlit-sales
Basic deployment of ML model (sales prediction) using **Streamlit** (python library).

## Execution
Command line (local): `streamlit streamlit_sales.py`  
Heroku: configuration set in `Procfile`. Application can be tested using [this link](https://oplx-py-streamlit-sales.herokuapp.com).

## Files information
- `sales_build_and_train.py` contains a sample generation of ML model (using sci-kit learn pipeline and pickle serialization).
- `streamlit_sales.py` is the base example of consumption of ML model (no HTML required, only use of **streamlit** functionality).
- `sales_generate_coreml.py` contains an in-progress generation of CoreML model (for internal use in apple coding).
- `requirements.txt`, `runtime.txt`, `Procfile` and `setup.sh` are used during web execution (Heroku).