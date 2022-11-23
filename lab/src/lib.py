# import libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, re, time, math, tqdm, itertools, json
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
colors = ["#03045e", "#023e8a", "#0077b6", '#0096c7', '#0096c7', '#00b4d8', '#90e0ef', '#ade8f4', '#caf0f8']

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

def make_seperate_datasets(cleaned_data, n_samples=20000, random_state=123):
    _cleaned_data = json.loads(cleaned_data['Label'].value_counts().to_json())

    _arrayX, _arrayY, _arrayResample, _count = [], [], [] , 0
    
    for it in _cleaned_data.keys():
        _data = cleaned_data[cleaned_data['Label'] == int(it)]
        _data_resample = resample(_data, n_samples=n_samples, random_state=random_state, replace=True)
        if _count == 0:
            _y = np.zeros(_data.shape[0])
        elif _count == 1:
            _y = np.ones(_data.shape[0])
        else:
            _y = np.full(_data.shape[0], 2)
        _y_begin, _count = pd.DataFrame(_y), _count + 1

        # Collect feature
        _arrayX.append(_data)
        _arrayY.append(_y_begin)
        _arrayResample.append(_data_resample)

    # merging the original dataframe
    return pd.concat(_arrayX, sort=True), pd.concat(_arrayY, sort=True), pd.concat(_arrayResample)

def Model(kernel_size=3, input_shape=(72, 1)):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', 
                    padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    
    # adding a pooling layer
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', 
                    padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Conv1D(filters=64, kernel_size=kernel_size, activation='relu', 
                    padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=(3), strides=2, padding='same'))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(6, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def main(path, x_header_networkdata = "Bwd Packets/s", y_header_networkdata = "min_seg_size_forward", columns_drop = ['Source Port','Flow ID','Source IP', 'Destination IP','Timestamp', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets', "PSH Flag Count","Init_Win_bytes_forward","Flow Bytes/s","Flow Packets/s", "Label"]):
    # Load dataset
    network_data = get_network_data(path)

    # Show information of network data
    print(network_data.shape)
    print('Number of Rows (Samples): %s' % str((network_data.shape[0])))
    print('Number of Columns (Features): %s' % str((network_data.shape[1])))

    print(network_data.columns)
    print(network_data.info())
    print(network_data['Label'].value_counts())

    _labels = list(json.loads(network_data['Label'].value_counts().to_json()).keys())

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
    print(cleaned_data['Label'].unique())

    X, y, train_dataset = make_seperate_datasets(cleaned_data)

    # Show information of X, y
    print(X.shape, '\n', y.shape)

    # checking if there are some null values in data
    X.isnull().sum().to_numpy()

    # viewing the distribution of intrusion attacks in our dataset 
    plt.figure(figsize=(10, 8))
    circle = plt.Circle((0, 0), 0.7, color='white')
    plt.title('Intrusion Attack Type Distribution')
    
    plt.pie(train_dataset['Label'].value_counts(), labels=_labels, colors=colors[:len(_labels)])
    p = plt.gcf()
    p.gca().add_artist(circle)
    plt.show()

    # Making X & Y Variables (CNN)
    test_dataset = train_dataset.sample(frac=0.1)
    target_train = train_dataset['Label']
    target_test = test_dataset['Label']
    print(target_train.unique(), target_test.unique())
    y_train = to_categorical(target_train, num_classes=6)
    y_test = to_categorical(target_test, num_classes=6)

    # Data Splicing
    train_dataset = train_dataset.drop(columns = columns_drop, axis = 1)
    test_dataset = test_dataset.drop(columns = columns_drop, axis = 1)

    # making train & test splits
    X_train = train_dataset.iloc[:, :-1].values
    X_test = test_dataset.iloc[:, :-1].values

    kernel_size = y_train.shape[1]
    input_shape = (X_train.shape[1], 1)

    model = Model(kernel_size=kernel_size, input_shape=input_shape)
    model.summary()
    logger = CSVLogger('logs.csv', append=True)
    his = model.fit(X_train, y_train, epochs=30, batch_size=0, 
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

def analyst(path=None):
    path = path if path else default_path
    if os.path.isfile(path):
        get_network_data(path)
    elif os.path.isdir(path):
        for dirname, _, filenames in os.walk(path):
            for filename in filenames:
                file = os.path.join(dirname, filename)
                print(f"We will show chart from {file}")
                try:
                    main(file)
                except Exception as err:
                    print(err)
    else:
        print("The path is not file or directory")


