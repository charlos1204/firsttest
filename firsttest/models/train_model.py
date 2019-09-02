import process_sequence_fasta as pro_seq_fasta
import sequence2vector as s2v_tools
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Dense
from keras.models import Sequential
from keras.layers import Embedding
from keras.preprocessing import sequence

"""
Author: Carlos Garcia-Perez
Date: 25.06.2019 final version of model setting.
                 add new option to train all data set no splitting.
                 save the model and the weight separated
      17.06.2019 fix the model definition
      14.06.2019 split data set in train, validation and test sets in option 1
      13.06.2019 create data set in one-hot-encoding and save as object.pkl in option 2
                 first version of the script
"""
opt = 2

type = 'nuc' # aa

if opt == 1:
    print('processing all...')
    x_data_name = '/data/sequence_dataset.pkl'
    y_data_name = '/data/label_dataset.pkl'

    X = pickle.load(open(x_data_name, 'rb'))
    Y = pickle.load(open(y_data_name, 'rb'))

    classes = pickle.load(open("/data/classes.pkl", 'rb'))

    print('defining model:')

    features = 20
    num_classes = classes

    print('features: ', features)
    print('clases: ', num_classes)
    print('nodes: ', 128)
    print('bacth size: ', 2000)
    print('epochs: ', 50)

    print('reshaping data...')
    max_len = max([len(s) for s in X])
    X_train = sequence.pad_sequences(X, maxlen=max_len)

    print('training dataset: ', X_train.shape)
    print('max_length:', max_len)

    model = Sequential()
    model.add(Embedding(len(X_train), features, input_length=max_len))
    model.add(LSTM(128))  # 32
    model.add(Dense(num_classes, activation='softmax'))

    print('compiling the model...')
    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('training the model...')
    # metric
    model.fit(X_train, Y,
              epochs=50,
              batch_size=2000)

    results_eval = model.evaluate(X_train, Y, batch_size=2000)

    print("%s: %.2f%%" % (model.metrics_names[1], results_eval[1] * 100))

    pickle.dump(results_eval, open("/data/results_eval.pkl", 'wb'), protocol=4)

    # serialize model to JSON
    model_json = model.to_json()
    with open("/data/model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/data/model.h5")
    print("Saved model to disk...")

elif opt == 2:
    print('loading data...')
    info = pickle.load(open("/data/info.pkl", 'rb'))
    X_train = pickle.load(open("/data/x_train.pkl", 'rb'))
    #X_test = pickle.load(open("/data/x_test.pkl", 'rb'))

    Y_train = pickle.load(open("/data/y_train.pkl", 'rb'))
    #Y_test = pickle.load(open("/data/y_test.pkl", 'rb'))

    print('defining model:')

    features = 20
    num_classes = info[0]
    max_len = info[1]
    ly = 128 # layer
    btch_size = 250
    epch = 50

    print('features: ', features)
    print('clases: ', num_classes)
    print('layer nodes: ', ly) # 128
    print('bacth size: ', btch_size) # 2000
    print('epochs: ', epch)

    print('reshaping data...')

    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    #X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    print('training dataset: ', X_train.shape)
    #print('test dataset: ', X_test.shape)
    print('max_length', max_len)

    model = Sequential()
    model.add(Embedding(len(X_train), features, input_length=max_len))
    model.add(LSTM(ly))  # 128
    model.add(Dense(num_classes, activation='softmax'))

    print('compiling the model...')
    # compile the model
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print('training the model...')
    # metric
    network = model.fit(X_train, Y_train,
                        epochs=epch,
                        batch_size=btch_size,
                        validation_split=0.2) # 0.2

    X_test = pickle.load(open("/data/x_test.pkl", 'rb'))
    Y_test = pickle.load(open("/data/y_test.pkl", 'rb'))
    
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    
    results_eval = model.evaluate(X_test, Y_test, batch_size=btch_size)

    print('training the model... done!!!')
    print('savinig the history...')
    pickle.dump(network, open("/data/history.pkl", 'wb'), protocol=4)
    pickle.dump(results_eval, open("/data/results_eval.pkl", 'wb'), protocol=4)
    print('done...')


elif opt == 3:
    x_data_name = '/data/sequence_dataset.pkl'
    y_data_name = '/data/label_dataset.pkl'

    X = pickle.load(open(x_data_name, 'rb'))
    Y = pickle.load(open(y_data_name, 'rb'))

    # spliting data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

    # saving the train, validation and test datasets
    pickle.dump(X_train, open("/data/x_train.pkl", 'wb'), protocol=4)
    pickle.dump(X_test, open("/data/x_test.pkl", 'wb'), protocol=4)
    pickle.dump(Y_train, open("/data/y_train.pkl", 'wb'), protocol=4)
    pickle.dump(Y_test, open("/data/y_test.pkl", 'wb'), protocol=4)

elif opt == 4:

    fname = '/data/subdataset_RDP_nucl.fasta'

    sequence_df = pro_seq_fasta.process_fasta(fname, type)

    Y = np.array(sequence_df['bacteria'])
    X = np.array(sequence_df['sequence'])

    max_len = max([len(s) for s in X])
    classes = len(np.unique(Y))
    info = (classes, max_len)

    Y = s2v_tools.label2one_hot_encoding(Y)

    pickle.dump(X, open("/data/sequence_dataset.pkl", 'wb'), protocol=4)
    pickle.dump(Y, open("/data/label_dataset.pkl", 'wb'), protocol=4)
    pickle.dump(info, open("/data/info.pkl", 'wb'), protocol=4)
