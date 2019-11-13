#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import dill as pickle
import os

# Constants for layout and dataframe manipulation
DATA_PATH = os.path.dirname(os.path.abspath(__file__)) + '/data/test_k.csv'
MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/model/model_v2.pk'
FEATURES = ['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
CHART = ['DayOfWeek', 'Customers', 'Sales', 'Promo']
HEAD_ROWS = 15
MIN_ROWS = 100
DEFAULT_ROWS = 1000
STEP_ROWS = 5000

# Data manipulation routines
@st.cache(persistent=True)
def load_data():
    loaded_data = pd.read_csv(DATA_PATH)
    loaded_data = loaded_data[FEATURES]
    return loaded_data
    
def load_model():
    pickle._dill._reverse_typemap['ClassType'] = type
    with open(MODEL_PATH, 'rb') as model:
        loaded_model = pickle.load(model)
    return loaded_model
    
def select_data(data, nrows, rndm):
    total_rows = len(data)
    if total_rows > nrows:
        drop_rows = total_rows - nrows
        if rndm == False:
            drop_indx = np.linspace(start=nrows, stop=total_rows - 1, 
                                    num=drop_rows, dtype=int)
        else:
            drop_indx = np.sort(np.random.choice(total_rows - 1, drop_rows, replace=False))
        selected_data = data.drop(drop_indx)
        selected_data.reset_index(inplace = True)
    else:
        selected_data = data
    return selected_data[FEATURES]

def predict_data(data, model, progress, progress_text):
    step_size = int(round(len(data) / 100 ))
    data_list = (data.loc[i:i+step_size-1,:] for i in range(0, len(data), step_size))
    predicted_values =  np.array([])
    progress_text.text('Initializing...')
    for step in data_list:
        predicted_step = model.predict(step)
        predicted_values = np.append(predicted_values, predicted_step)
        percentage = len(predicted_values) / len(data) * 100
        text = "Processed {} of {} registries ({:.0f}%)".format(
                len(predicted_values), len(data), percentage)
        progress_text.text(text)
        progress.progress(int(round(percentage)))
    data['Sales'] = predicted_values
    return predicted_values, data
    
def highlight_data(series, metric):
    bg_color = 'CCFFE5' if series['Sales'] > metric else 'FFCCCC'
    return ['background-color: #{}'.format(bg_color)]*len(series)
        
def get_chart_data(data):
    altchart = alt.Chart(data[CHART]).mark_circle().encode(
            alt.X(alt.repeat("column"), type='quantitative'),
            alt.Y(alt.repeat("row"), type='quantitative'),
            color='Promo:N'
        ).properties(
            width=150,
            height=150
        ).repeat(
            row=['DayOfWeek', 'Customers', 'Sales'],
            column=['Sales', 'Customers', 'DayOfWeek']
        ).interactive()
    return st.altair_chart(altchart, width=-1)

# Data loading (and persistence)
with st.spinner('Generating Scatter Matrix...'):
    loaded_data = load_data()

# Sidebar layout (filters for dataframe and prediction)
__side_title = st.sidebar.title('Filters')
__side_option = st.sidebar.subheader('Type of information:')
__side_select = st.sidebar.selectbox(
        'Select information to display',
        ('Sales data detail','Sales amount prediction detail'))

# Main page layout (dataframe and related elements)
__main_title = st.title('Sales predictor')
__main_subtitle = st.empty()
__main_data_pstat = st.empty()
__main_predictor_pstat = st.empty()

# Main logic to update screen elements (sidebar and main page)
if __side_select == 'Sales data detail':
    # Show sidebar filters for dataframe
    __side_data_size = st.sidebar.radio(
            'Data frame rows to show:',
            ('Only header', 'All information (slow)'))
    __side_data_exec = st.sidebar.button('Show data!')
    if __side_data_exec:
        # When "Show data!" is clicked, show dataframe information at main page
        __main_subtitle = st.subheader('Sales data detail')
        __main_data_descr = st.dataframe(loaded_data.describe())
        if __side_data_size == 'Only header':
            with st.spinner('Loading header ({} registries)...'.format(HEAD_ROWS)):
                __main_data_frame = st.dataframe(loaded_data.head(HEAD_ROWS))
        else:
            with st.spinner('Loading {} registries for visualization...'.format(len(loaded_data))):
                __main_data_frame = st.dataframe(loaded_data)
    else:
        __main_data_pstat.warning('Sales data pending... click Show data!')
else:
    # Show sidebar information for prediction
    __side_predictor_rows = st.sidebar.slider('Number of registries to process', 
                                 MIN_ROWS, 
                                 len(loaded_data),
                                 MIN_ROWS, 
                                 STEP_ROWS)
    __side_predictor_rand = st.sidebar.checkbox('Random selection')
    __side_predictor_exec = st.sidebar.button('Run!')
    if __side_predictor_exec:
        # When "Run!" is clicked, show prediction information at main page
        __main_subtitle = st.subheader('Sales amount prediction detail')
        __main_predictor_pstat.info('Selecting sales to process...')
        selected_data = select_data(loaded_data, 
                                    __side_predictor_rows, 
                                    __side_predictor_rand)
        __main_predictor_frame = st.dataframe(selected_data)
        __main_predictor_pstat.info('Predicting sales amount...')
        __main_predictor_prbar = st.empty()
        __main_predictor_prtxt = st.empty()
        loaded_model = load_model()
        predicted_values, selected_data = predict_data(selected_data, 
                                                       loaded_model, 
                                                       __main_predictor_prbar, 
                                                       __main_predictor_prtxt)
        metric = sum(predicted_values) / len(predicted_values)
        __main_predictor_pstat.info('Highlighting sales prediction results...')
        with st.spinner('Evaluating sales based on average...'):
            __main_predictor_frame.dataframe(selected_data.style.apply(
                    highlight_data, metric=metric, axis=1))
        __main_predictor_pstat.info('Generating sales prediction chart...')
        with st.spinner('Generating Scatter Matrix...'):
            __main_predictor_chart = get_chart_data(selected_data)
        __main_predictor_pstat.success('Sales amount predicted!')
    else:
        __main_predictor_pstat.warning('Sales prediction pending... click Run!')

