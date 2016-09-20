import pandas as pd
import numpy as np
from sklearn import datasets

# example for loading a dataset from scikit-learn
boston_data = datasets.load_boston()

# example of loading data on the local host using pandas
# index_col = # if a specific column (specified by its position, #) should be used as the index (1,2,3,...)
# header=None if there are no column headers
# sep = ',' for comma sperated, sep=' ' for space seperated, etc...
data = pd.read_csv('/path/to/data.csv')

# for the boston data it's a bit different, first look at the keys
# the keys for this dataset are: 'data', 'feature_names', 'DESCR', 'target'
# 'feature_names' is equivalent to column headers
print(boston_data.keys())
boston_features = boston['data']
boston_response = boston['target']
boston_feature_names = boston['feature_names']

# look at some of the data
data.head()

# data is a pandas dataframe while boston_features is a numpy array
# for this reason, boston_features.head() won't work
# instead we turn it, along with the feature_names into a pandas dataframe
boston_df = pd.DataFrame(data=boston_features, columns=boston_feature_names)
print(boston_df.head())

# add a column to the dataframe, 
# unless a position is specified column gets added to the end
boston_df['target'] = boston_data.target

# look at the columns headers from local data
print(data.columns)

# access specific column of data
data['column_header'] 
# or
data.column_header

# quickly computer summary statistics on boston_df
boston_df.describe()

# compute the correlation matrix of boston_df
boston_df.corr()
