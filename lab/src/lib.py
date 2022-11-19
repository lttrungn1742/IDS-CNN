# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, re, time, math, tqdm, itertools
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from sklearn.utils import resample

default_path = os.getenv('CNN_LAB_PATH',"/Users/TrungLT/personal/CNN/Input")

def sanitize(column):
    while ' ' == column[0]:
        column = column[1:]
    return column

def loading_data(path):
    """
      https://stackoverflow.com/questions/45529507/unicodedecodeerror-utf-8-codec-cant-decode-byte-0x96-in-position-35-invalid
    """
    return pd.read_csv(path, encoding='cp1252')

def validate(network_data):
    try:
        network_data['Label']
    except KeyError:
        columns = [sanitize(i) for i in network_data.columns]
        network_data.columns = columns

def get_network_data(path):
    network_data = loading_data(path)
    validate(network_data)
    return network_data
    
def main(path, x_header_networkdata="Bwd Packets/s", y_header_networkdata="min_seg_size_forward"):
    # Load dataset
    network_data = get_network_data(path)
    
    # Show information of network data
    print(network_data.shape)
    print('Number of Rows (Samples): %s' % str((network_data.shape[0])))
    print('Number of Columns (Features): %s' % str((network_data.shape[1])))
    network_data.head(4)
    print(network_data.columns)
    print(network_data.info())
    print(network_data['Label'].value_counts())

    sns.set(rc={'figure.figsize':(12, 6)})
    plt.xlabel('Attack Type')
    sns.set_theme()
    ax = sns.countplot(x='Label', data=network_data)
    ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
    plt.show()

    pyo.init_notebook_mode()
    fig = px.scatter(x = network_data[x_header_networkdata][:100000], 
                    y=network_data[y_header_networkdata][:100000])
    fig.show()

    sns.set(rc={'figure.figsize':(12, 6)})
    sns.scatterplot(x=network_data[x_header_networkdata][:50000],
                     y=network_data[y_header_networkdata][:50000], 
                hue='Label', data=network_data)

    # check the dtype of timestamp column
    (network_data['Timestamp'].dtype)

    # check for some null or missing values in our dataset
    network_data.isna().sum().to_numpy()

    # drop null or missing columns
    cleaned_data = network_data.dropna()
    cleaned_data.isna().sum().to_numpy()

    # encode the column labels
    label_encoder = LabelEncoder()
    cleaned_data['Label']= label_encoder.fit_transform(cleaned_data['Label'])
    cleaned_data['Label'].unique()

        # check for encoded labels
    cleaned_data['Label'].value_counts()

def handler():
    path = default_path
    if os.path.isfile(path):
        get_network_data(path)
    elif os.path.isdir(path):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                file = os.path.join(dirname, filename)
                print(f"We will show chart from {file}")
                try:
                    get_network_data(file)
                except Exception as err:
                    print(err)
    else:
        print("The path is not file or directory")


