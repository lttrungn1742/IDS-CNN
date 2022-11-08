import numpy as np
import pandas as pd 
import os, re, time, math, tqdm, itertools, sys
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
import keras
from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.utils import resample

def _scatter_plot(network_data, vertical_axis_label, horizontal_axis_label):
    pyo.init_notebook_mode(connected=True)
    fig = px.scatter(x=network_data[vertical_axis_label][:100000],
                     y=network_data[horizontal_axis_label][:100000])
    fig.show()

    sns.set(rc={'figure.figsize':(12, 6)})
    plt.xlabel(horizontal_axis_label)
    sns.set_theme()
    ax = sns.scatterplot(x=network_data[vertical_axis_label][:50000], y=network_data[horizontal_axis_label][:50000], 
                hue='Label', data=network_data)
    ax.set(xlabel=vertical_axis_label, ylabel=horizontal_axis_label)   
    plt.show()



def scatter_plot(path="/input", isShowInfomation=False, vertical_axis_label="Bwd Packets/s", horizontal_axis_label="Fwd Packet Length Min"):
    if os.path.isfile(path):
        network_data = get_network_data(path, isShowInfomation=isShowInfomation)
        _scatter_plot(network_data, vertical_axis_label, horizontal_axis_label)

    elif os.path.isdir(path):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                file = os.path.join(dirname, filename)
                print(f"-  {file}")
                try:
                    network_data = get_network_data(file, isShowInfomation=isShowInfomation)
                    _scatter_plot(network_data, vertical_axis_label, horizontal_axis_label)
                except Exception as err:
                    print(err)
    else:
        print("The path is not file or directory")

def loading_data(path):
    """
      https://stackoverflow.com/questions/45529507/unicodedecodeerror-utf-8-codec-cant-decode-byte-0x96-in-position-35-invalid
    """
    return pd.read_csv(path, encoding='cp1252')

def validate(network_data):
    """
        ptr
    """
    try:
        network_data['Label']
    except KeyError:
        columns = [sanitize(i) for i in network_data.columns]
        network_data.columns = columns

def sanitize(column):
    """
        de-recursion
    """
    while ' ' == column[0]:
        column = column[1:]
    return column

def _plot_number(network_data):
    sns.set(rc={'figure.figsize': (12, 6)})
    plt.xlabel('Attack Type')
    sns.set_theme()
    ax = sns.countplot(x='Label', data=network_data)
    ax.set(xlabel='Attack Type', ylabel='Number of Attacks')
    plt.show()

def plot_number(path="/input/", isShowInfomation=False):
    if os.path.isfile(path):
        network_data = get_network_data(path, isShowInfomation=isShowInfomation)
        _plot_number(network_data)

    elif os.path.isdir(path):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                file = os.path.join(dirname, filename)
                print(f"-  {file}")
                try:
                    network_data = get_network_data(file, isShowInfomation=isShowInfomation)
                    _plot_number(network_data)
                except Exception as err:
                    print(err)
    else:
        print("The path is not file or directory")

def _circle(network_data):
    cleaned_data = network_data.dropna()
    cleaned_data.isna().sum().to_numpy()
    label_encoder = LabelEncoder()
    cleaned_data['Label']= label_encoder.fit_transform(cleaned_data['Label'])
    cleaned_data['Label'].unique()
    cleaned_data['Label'].value_counts()
    data_1 = cleaned_data[cleaned_data['Label'] == 0]
    data_2 = cleaned_data[cleaned_data['Label'] == 1]
    data_3 = cleaned_data[cleaned_data['Label'] == 2]

    # make benign feature
    y_1 = np.zeros(data_1.shape[0])
    y_benign = pd.DataFrame(y_1)

    # make bruteforce feature
    y_2 = np.ones(data_2.shape[0])
    y_bf = pd.DataFrame(y_2)

    # make bruteforceSSH feature
    y_3 = np.full(data_3.shape[0], 2)
    y_ssh = pd.DataFrame(y_3)

    # merging the original dataframe
    X = pd.concat([data_1, data_2, data_3], sort=True)
    y = pd.concat([y_benign, y_bf, y_ssh], sort=True)
    data_1_resample = resample(data_1, n_samples=20000, random_state=123, replace=True)
    data_2_resample = resample(data_2, n_samples=20000, random_state=123, replace=True)
    data_3_resample = resample(data_3, n_samples=20000, random_state=123, replace=True)
    train_dataset = pd.concat([data_1_resample, data_2_resample, data_3_resample])
    train_dataset.head(2)
    plt.figure(figsize=(10, 8))
    circle = plt.Circle((0, 0), 0.7, color='white')
    plt.title('Intrusion Attack Type Distribution')
    plt.pie(train_dataset['Label'].value_counts(), labels=['Benign', 'BF', 'BF-SSH'], colors=['blue', 'magenta', 'cyan'])
    p = plt.gcf()
    p.gca().add_artist(circle)

    test_dataset = train_dataset.sample(frac=0.1)
    target_train = train_dataset['Label']
    target_test = test_dataset['Label']
    target_train.unique(), target_test.unique()

    y_train = to_categorical(target_train, num_classes=3)
    y_test = to_categorical(target_test, num_classes=3)

    train_dataset = train_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)
    test_dataset = test_dataset.drop(columns = ["Timestamp", "Protocol","PSH Flag Cnt","Init Fwd Win Byts","Flow Byts/s","Flow Pkts/s", "Label"], axis=1)

    X_train = train_dataset.iloc[:, :-1].values
    X_test = test_dataset.iloc[:, :-1].values
    X_test.show()

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    # reshape the data for CNN
    X_train = X_train.reshape(len(X_train), X_train.shape[1], 1)
    X_test = X_test.reshape(len(X_test), X_test.shape[1], 1)
    X_train.shape, X_test.shape
    model = Model()
    model.summary()
    logger = CSVLogger('logs.csv', append=True)
    his = model.fit(X_train, y_train, epochs=30, batch_size=32, 
          validation_data=(X_test, y_test), callbacks=[logger])
    scores = model.evaluate(X_test, y_test)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    history = his.history
    history.keys()
    epochs = range(1, len(history['loss']) + 1)
    acc = history['accuracy']
    loss = history['loss']
    val_acc = history['val_accuracy']
    val_loss = history['val_loss']

    # visualize training and val accuracy
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Accuracy (CNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(epochs, acc, label='accuracy')
    plt.plot(epochs, val_acc, label='val_acc')
    plt.legend()

    # visualize train and val loss
    plt.figure(figsize=(10, 5))
    plt.title('Training and Validation Loss(CNN)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='loss', color='g')
    plt.plot(epochs, val_loss, label='val_loss', color='r')
    plt.legend()

def circle(path="/input/", isShowInfomation=False):
    if os.path.isfile(path):
        network_data = get_network_data(path, isShowInfomation=isShowInfomation)
        _circle(network_data)

    elif os.path.isdir(path):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                file = os.path.join(dirname, filename)
                print(f"-  {file}")
                try:
                    network_data = get_network_data(file, isShowInfomation=isShowInfomation)
                    _circle(network_data)
                except Exception as err:
                    print(err)
    else:
        print("The path is not file or directory")    

def Model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=6, activation='relu', 
                    padding='same', input_shape=(72, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_network_data(path, isShowInfomation=True):
    network_data = loading_data(path)
    validate(network_data)
    if isShowInfomation:
        print(network_data.shape)
        print(network_data.info())
        print(network_data['Label'].value_counts())
    return network_data
    

def handler():
    path = "/input"
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


