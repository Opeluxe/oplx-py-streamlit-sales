import pandas as pd
import lightgbm as lgb
import os
import dill as pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")


def build_and_train(train_path):
    data = pd.read_csv(train_path, sep=',', error_bad_lines=False, low_memory=False, nrows=30000)
    
    features = ['Store', 'DayOfWeek', 'Date', 'Customers', 'Open', 'Promo', 'StateHoliday', 'SchoolHoliday']
    X_train, X_test, y_train, y_test = train_test_split(data[features], data['Sales'], test_size=0.25, random_state=42)
    
    pipe = make_pipeline(PreProcessing(), lgb.LGBMRegressor())
    
    param_grid = {"lgbmregressor__learning_rate": [0.1, 0.2],
                  "lgbmregressor__max_depth": [3, 5],
                  "lgbmregressor__min_child_samples": [20, 100],
                  "lgbmregressor__num_leaves": [8, 10, 20],
                  "lgbmregressor__objective": ['regression']}
    
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    
    grid.fit(X_train, y_train)
    
    return(grid)


class PreProcessing(BaseEstimator, TransformerMixin):
    """Custom Pre-Processing estimator for our use-case
    """

    def __init__(self):
        pass

    def transform(self, df):
        """Regular transform() that is a help for training, validation & testing datasets
        """
        df['size_date'] = df.groupby('Date')['Store'].transform('size')
        df['total_clients_per_date'] = df.groupby('Date')['Customers'].transform('sum')
        df['Date'] = pd.to_datetime(df['Date'],yearfirst=True)
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['year'] = df['Date'].dt.year
        df['customer_per_month_in_store'] = df.groupby(['Store','month','year'])['Customers'].transform('sum')
        df['store_size_month'] = df.groupby(['Store','month','year'])['Store'].transform('size')
        df['StateHoliday'] = self.categoricalValues(df, 'StateHoliday')
        df['Store'] = self.categoricalValues(df, 'Store')
        df = df.drop('Date', axis=1)
        return df.as_matrix()

    def fit(self, df, y=None, **fit_params):
        """Fitting the Training dataset & calculating the required values from train
        """
        #self.customers_mean_ = df['Customerss'].mean()
        return self
    
    def categoricalValues(self, df, column):
        s = pd.Categorical(df[column])
        s = s.codes
        s.astype(int)
        return s

if __name__ == '__main__':
#   Build predictor (regression model)
    base_path = os.path.dirname(os.path.abspath("__file__"))
    train_file = base_path + '/data/train_k.csv'
    model = build_and_train(train_file)

#   Store the model
    filename = base_path + '/model/model_v2.pk'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

