import tensorflow as tf

def simpleNN(X):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(
        128, activation='relu', input_shape=X.shape[1:]))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def simpleLSTM(X):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(8, input_shape=X.shape[1:]))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(loss='binary_crossentropy',
                  optimizer=opt, metrics=['accuracy'])
    return model
