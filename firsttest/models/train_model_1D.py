import sys
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Conv1D, Dense
from keras.layers import MaxPooling1D, GlobalMaxPooling1D
from keras.preprocessing import sequence
import sequence2vector as s2v_tools
import process_sequence_fasta as pro_seq_fasta
from sklearn.model_selection import train_test_split


"""
Author: Carlos Garcia-Perez
Date: 26.06.2019 1D CNN for sequence classification
                 first version of the script
"""

# settings
features = 20
ly = 128 # layer
btch_size = 10
epch = 20

opt = 1 # int(sys.argv[1])
type = 'nuc'
print('running option = ', opt)


if opt == 1:
    print('loading data...')
    info = pickle.load(open("/srv/firsttest/data/info.pkl", 'rb'))
    X_train = pickle.load(open("/srv/firsttest/data/x_train.pkl", 'rb'))
    #X_test = pickle.load(open("/data/x_test.pkl", 'rb'))

    Y_train = pickle.load(open("/srv/firsttest/data/y_train.pkl", 'rb'))
    #Y_test = pickle.load(open("/data/y_test.pkl", 'rb'))

    print('defining model:')

    #features = 20
    num_classes = info[0]
    max_len = info[1]

    print('features: ', features)
    print('clases: ', num_classes)
    print('layer : ', ly)
    print('bacth size: ', btch_size)
    print('epochs: ', epch)

    print('reshaping data...')

    #X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    #X_test = sequence.pad_sequences(X_test, maxlen=max_len)

    print('training dataset: ', X_train.shape)
    #print('test dataset: ', X_test.shape)
    print('max_length', max_len)

    model = Sequential()
    model.add(Embedding(len(X_train), features, input_length=max_len))
    model.add(Conv1D(ly, 9, activation='relu'))
    model.add(MaxPooling1D(5))
    model.add(Conv1D(ly, 9, activation='relu'))
    model.add(GlobalMaxPooling1D())
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
                        validation_split=0.2)


    X_test = pickle.load(open("/srv/firsttest/data/x_test.pkl", 'rb'))
    Y_test = pickle.load(open("/srv/firsttest/data/y_test.pkl", 'rb'))

    #X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    results_eval = model.evaluate(X_test, Y_test, batch_size=btch_size)

    print('training the model... done!!!')
    #print('savinig the history...')
    #pickle.dump(network, open("/data/history_1D.pkl", 'wb'), protocol=4)
    #pickle.dump(results_eval, open("/data/results_eval_1D.pkl", 'wb'), protocol=4)
    
    # serialize model to JSON
    model_json = model.to_json()
    with open("/srv/firsttest/model_trained/model_1D.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("/srv/firsttest/model_trained/model_1D.h5")
    print("Saved model to disk...")
    print('done...')

elif opt == 2:
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

elif opt == 3:

    #fname = '/data/RDP_sequences_filtered_nucl2.fasta'
    fname = '/data/paper_dataset.fasta'
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

