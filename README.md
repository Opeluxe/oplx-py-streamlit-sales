# oplx-py-streamlit-sales
Basic use of ML model (sales prediction) using **Streamlit** (python library).

## Execution
Command line (local): `streamlit streamlit_sales.py`  
Heroku: configuration set in `Procfile`. Web app can be tested using [this link](https://oplx-py-streamlit-sales.herokuapp.com).

## Files information
- `sales_build_and_train.py` contains a simple generation of ML model (using sci-kit learn pipeline and pickle serialization).
- `streamlit_sales.py` is a basic example of ML model consumption (no HTML required, only use of **streamlit** functionality, no need to deploy as web service).
- `sales_generate_coreml.py` contains an in-progress implementation of CoreML model (for internal use in apple-swift coding).
- `requirements.txt`, `runtime.txt`, `Procfile` and `setup.sh` are used during web execution (Heroku).