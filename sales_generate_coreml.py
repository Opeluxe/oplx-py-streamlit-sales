# -*- coding: utf-8 -*-
import coremltools
import dill as pickle
import os

MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + '/model/model_v2.pk'
FEATURES = ['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']

def load_model():
    pickle._dill._reverse_typemap['ClassType'] = type
    with open(MODEL_PATH, 'rb') as model:
        loaded_model = pickle.load(model)
    return loaded_model

def convert_model(model):
#   Convert to coreml model
    coreml_model = coremltools.converters.sklearn.convert(model, FEATURES, "Sales")
    
#   Set model metadata
    coreml_model.author = 'Opeluxe'
    coreml_model.license = 'BSD'
    coreml_model.short_description = 'Predicts the sales amount in specific stores.'

#   Set feature descriptions manually
    coreml_model.input_description['Store'] = 'Store ID'
    coreml_model.input_description['DayOfWeek'] = 'Day of the sale'
    coreml_model.input_description['Date'] = 'Date of the sale'
    coreml_model.input_description['Customers'] = 'Number of customers in the store'
    coreml_model.input_description['Open'] = 'Store open'
    coreml_model.input_description['Promo'] = 'Promo applies that date'
    coreml_model.input_description['StateHoliday'] = 'State Holiday'
    coreml_model.input_description['SchoolHoliday'] = 'School Holiday'

#   Set the output descriptions
    coreml_model.output_description['Sales'] = 'Predicted amount'

#   Save the model
    coreml_model.save('SalesML.mlmodel')

if __name__ == '__main__':
#   Recover model
    model = load_model()
#   Convert and save the model
    convert_model(model)