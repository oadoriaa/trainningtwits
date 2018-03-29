from matplotlib import pylab as plt
import h5py
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model, model_from_json
from keras.optimizers import Adam


def saving_model_into_disk(model, name='model'):
    # serialize model to JSON
    model_json = model.to_json()
    with open(name + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(name + ".h5")
    print("Saved %s to disk" % name)


def load_model_from_disk(name='model'):
    # load json and create model
    json_file = open(name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(name + ".h5")
    print("Loaded model from disk")
    return loaded_model


def get_model_results(history, name='model'):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = [i for i in range(1, len(acc) + 1)]
    print(('acc = %0.2f, loss = %0.2f ') % (max(acc), min(loss)))
    print(('val_acc = %0.2f, val_loss = %0.2f ') % (max(val_acc), min(val_loss)))

    f1 = plt.figure()
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    f2 = plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    ## Saving
    f1.savefig(name + "_accuracy.pdf", bbox_inches='tight')
    f2.savefig(name + "_loss.pdf", bbox_inches='tight')


def load_embedding_matrix(h5filename='embedding_matrix_twitter.h5'):
    """Loading Embedded Matrix from a h5 file"""
    h5_embedded = h5py.File(h5filename, 'r')
    embedding_matrix = h5_embedded['embedding_matrix'][:]
    h5_embedded.close()
    return embedding_matrix


def load_twitter_data(h5filename='data_twitter.h5'):
    """Read twitts database to train from a h5 file. This is related
    to the embedding file."""
    h5_data_twitter = h5py.File(h5filename, 'r')
    data = h5_data_twitter['data'][:]
    labels = h5_data_twitter['labels'][:]
    h5_data_twitter.close()
    return data, labels


def get_train_test_datasets(data, labels):
    """ Split data of observables and label to train data"""
    #  training_samples = 100000
    #  validation_samples = 10000

    X_train, X_val, Y_train, Y_val = train_test_split(data, labels,
                                                      test_size=0.1,
                                                      # train_size = 0.8,
                                                      random_state=1234,
                                                      shuffle=True,
                                                      stratify=labels)
    return X_train, X_val, Y_train, Y_val


def model_twitter_convnet_lstm(embedding_matrix, maxlen):
    input_twitter = layers.Input(shape=(maxlen,), dtype='int32', name='input_twitter')
    x = layers.Embedding(embedding_matrix.shape[0],
                         embedding_matrix.shape[1],
                         input_length=maxlen,
                         trainable=True,
                         weights=[embedding_matrix],
                         mask_zero=False,
                         name='embedded')(input_twitter)

    x = layers.Conv1D(32, (5), padding='same', kernel_initializer='orthogonal', name='conv1d_1')(x)
    x = layers.Activation('elu', name='act_elu_1')(x)
    x = layers.MaxPool1D(3, name='maxpool_1')(x)
    x = layers.Dropout(0.25, name='dropout_1')(x)

    x = layers.Conv1D(32, (5), padding='same', kernel_initializer='orthogonal', name='conv1d_2')(x)
    x = layers.Activation('elu', name='act_elu_2')(x)
    x = layers.MaxPool1D(3, name='maxpool_2')(x)
    x = layers.Dropout(0.25, name='dropout_2')(x)

    x = layers.Bidirectional(layers.LSTM(32,
                                         return_sequences=True,
                                         kernel_initializer='orthogonal',
                                         name='lstm_1'))(x)
    x = layers.Activation('elu', name='act_elu_3')(x)
    x = layers.Bidirectional(layers.LSTM(64,
                                         return_sequences=False,
                                         kernel_initializer='orthogonal',
                                         name='lstm_2'))(x)
    x = layers.Activation('elu', name='act_elu_4')(x)

    x = layers.Dense(1, name='dense_final', activation='sigmoid')(x)

    model = Model(input_twitter, x)
    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


def model_twitter_convnet(embedding_matrix, maxlen):
    input_twitter = layers.Input(shape=(maxlen,), dtype='int32', name='input_twitter')
    x = layers.Embedding(embedding_matrix.shape[0],
                         embedding_matrix.shape[1],
                         input_length=maxlen,
                         trainable=True,
                         weights=[embedding_matrix],
                         mask_zero=False,
                         name='embedded')(input_twitter)

    x = layers.Conv1D(32, (3), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_1')(x)
    x = layers.Conv1D(32, (3), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_2')(x)
    x = layers.Conv1D(32, (3), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_3')(x)
    x = layers.Conv1D(32, (3), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_4')(x)
    x = layers.MaxPool1D(2, name='maxpool_2')(x)
    x = layers.Dropout(0.5, name='dropout_1')(x)

    # x = layers.Conv1D(32, (2), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_5')(x)
    # x = layers.Conv1D(32, (2), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_6')(x)
    # x = layers.Conv1D(32, (2), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_7')(x)
    # x = layers.Conv1D(32, (2), activation='elu', padding='same', kernel_initializer='orthogonal', name='conv1d_8')(x)
    # x = layers.Dropout(0.25, name = 'dropout_2')(x)

    x = layers.Flatten(name='flatten_1')(x)

    x = layers.Dense(128, activation='tanh')(x)
    x = layers.Dropout(0.5, name='dropout_3')(x)

    x = layers.Dense(128, activation='tanh')(x)
    x = layers.Dropout(0.5, name='dropout_4')(x)

    x = layers.Dense(1, name='dense_final', activation='sigmoid')(x)

    model = Model(input_twitter, x)
    model.compile(optimizer=Adam(lr=0.0001, decay=1e-6), loss='binary_crossentropy', metrics=['acc'])
    print(model.summary())
    return model


if __name__ == '__main__':
    ## Setting up names
    network_name = 'conv_smaller_3'  # 'conv_lstm'
    sufix = '10000_100'
    network_final_name = network_name + '_' + sufix

    ## Load Data
    embedding_matrix = load_embedding_matrix(h5filename='embedding_matrix_' + sufix + '_twitter.h5')
    maxlen = 32
    model = model_twitter_convnet(embedding_matrix, maxlen)
    data, labels = load_twitter_data()
    X_train, X_val, Y_train, Y_val = get_train_test_datasets(data, labels)

    ## Train Network
    history = model.fit(X_train,
                        Y_train,
                        epochs=10,
                        batch_size=64,
                        validation_data=(X_val, Y_val),
                        verbose=True)

    ## Save Network
    saving_model_into_disk(model, network_final_name)

    ## Ploting results
    get_model_results(history, network_final_name)
