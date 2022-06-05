import tensorflow as tf

def buildDeeperModel(embeddingMatrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_DIM, DROPOUT, NUM_CLASSES, LEARNING_RATE):
    embeddingLayer = tf.keras.layers.Embedding(embeddingMatrix.shape[0], EMBEDDING_DIM, weights=[embeddingMatrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    model = tf.keras.models.Sequential()
    model.add(embeddingLayer)
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_DIM)))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid'))

    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model

def buildModel(embeddingMatrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_DIM, DROPOUT, NUM_CLASSES, LEARNING_RATE):
    """
    Constructs the architecture of the model
    Input:
        embeddingMatrix : The embedding matrix to be loaded in the embedding layer.
    Output:
        model : A basic LSTM model
    """
    embeddingLayer = tf.keras.layers.Embedding(embeddingMatrix.shape[0], EMBEDDING_DIM, weights=[embeddingMatrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False)
    model = tf.keras.models.Sequential()
    model.add(embeddingLayer)
    model.add(tf.keras.layers.LSTM(LSTM_DIM, dropout=DROPOUT))
    model.add(tf.keras.layers.Dense(NUM_CLASSES, activation='sigmoid'))

    rmsprop = tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['acc'])
    return model