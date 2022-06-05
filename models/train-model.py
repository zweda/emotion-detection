import io
import json
import os
import sys

import numpy as np
import tensorflow as tf
from prepocessing import label2emotion

# ----------------------------------- #

from prepocessing import preprocessEmojiData as preprocessData
from build import buildDeeperModel as buildModel
MODEL_NAME = 'bilstm-twitter-emoji-replaced-EP%d_LR%de-5_LDim%d_BS%d'
CONFIG_FILE = 'baseline.config'
GLOVE_FILE = 'glove.twitter.27B.200d.txt'

# ----------------------------------- #

TRAIN_DATA_PATH = ""  # Path to training data file.
TEST_DATA_PATH = ""  # Path to testing data file.
SOLUTION_DIR = ""  # Output file that will be generated. This file can be directly submitted.
GLOVE_DIR = ""  # Path to directory where GloVe file is saved.
SAVED_MODELS_DIR = 'saved'
CONFIG_DIR = 'configs'
LOG_DIR = 'logs'

NUM_FOLDS = 0  # Value of K in K-fold Cross Validation
NUM_CLASSES = 0  # Number of classes - HAPPY, SAD, ANGRY, OTHERS
MAX_NB_WORDS = 0  # To set the upper limit on the number of tokens extracted using keras.preprocessing.text.Tokenizer
MAX_SEQUENCE_LENGTH = 0  # All sentences having lesser number of words than this will be padded
EMBEDDING_DIM = 0  # The dimension of the word embeddings
BATCH_SIZE = 0  # The batch size to be chosen for training the model.
LSTM_DIM = 0  # The dimension of the representations learnt by the LSTM model
NUM_EPOCHS = 0  # Number of epochs to train a model for

DROPOUT = 0. # Fraction of the units to drop for the linear transformation of the inputs.
LEARNING_RATE = 0.  # Learning rate of a model

LOGGER = None
def log(line, args=None, close=False):
    """
    Util function used to capture training metrics.
    Input:
        line : string that goes into a log file and is printed in console.
        args : additional arguments that are forwarded to the output stream.
    """
    if args is not None:
        print(line, args)
    else:
        print(line)

    global LOGGER
    if LOGGER is None:
        logfile = (MODEL_NAME + '.txt') % (NUM_EPOCHS, int(LEARNING_RATE * (10 ** 5)), LSTM_DIM, BATCH_SIZE)
        LOGGER = io.open(os.path.join(sys.path[0], LOG_DIR, logfile), 'w', encoding='utf8')
    LOGGER.write(line)
    if args is not None:
        LOGGER.write(str(args))
    LOGGER.write('\n')

    if close:
        LOGGER.close()
        LOGGER = None


def getMetrics(predictions, ground):
    """
    Given predicted labels and the respective ground truth labels, display some metrics
    Input: shape [# of samples, NUM_CLASSES]
        predictions : Model output. Every row has 4 decimal values, with the highest belonging to the predicted class
        ground : Ground truth labels, converted to one-hot encodings. A sample belonging to Happy class will be [0, 1, 0, 0]
    Output:
        accuracy : Average accuracy
        microPrecision : Precision calculated on a micro level. Ref - https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin/16001
        microRecall : Recall calculated on a micro level
        microF1 : Harmonic mean of microPrecision and microRecall. Higher value implies better classification  
    """
    # [0.1, 0.3 , 0.2, 0.1] -> [0, 1, 0, 0]
    discretePredictions = tf.keras.utils.to_categorical(predictions.argmax(axis=1))

    truePositives = np.sum(discretePredictions * ground, axis=0)
    falsePositives = np.sum(np.clip(discretePredictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - discretePredictions, 0, 1), axis=0)

    log("True Positives per class : ", truePositives)
    log("False Positives per class : ", falsePositives)
    log("False Negatives per class : ", falseNegatives)

    # ------------- Macro level calculation ---------------
    macroPrecision = 0
    macroRecall = 0
    # We ignore the "Others" class during the calculation of Precision, Recall and F1
    for c in range(1, NUM_CLASSES):
        precision = truePositives[c] / (truePositives[c] + falsePositives[c])
        macroPrecision += precision
        recall = truePositives[c] / (truePositives[c] + falseNegatives[c])
        macroRecall += recall
        f1 = (2 * recall * precision) / (precision + recall) if (precision + recall) > 0 else 0
        log("Class %s : Precision : %.3f, Recall : %.3f, F1 : %.3f" % (label2emotion[c], precision, recall, f1))

    macroPrecision /= 3
    macroRecall /= 3
    macroF1 = (2 * macroRecall * macroPrecision) / (macroPrecision + macroRecall) if (macroPrecision + macroRecall) > 0 else 0
    log("Ignoring the Others class, Macro Precision : %.4f, Macro Recall : %.4f, Macro F1 : %.4f" % (macroPrecision, macroRecall, macroF1))

    # ------------- Micro level calculation ---------------
    truePositives = truePositives[1:].sum()
    falsePositives = falsePositives[1:].sum()
    falseNegatives = falseNegatives[1:].sum()

    log("Ignoring the Others class, Micro TP : %d, FP : %d, FN : %d" % (truePositives, falsePositives, falseNegatives))

    microPrecision = truePositives / (truePositives + falsePositives)
    microRecall = truePositives / (truePositives + falseNegatives)

    microF1 = (2 * microRecall * microPrecision) / (microPrecision + microRecall) if (microPrecision + microRecall) > 0 else 0
    # -----------------------------------------------------

    predictions = predictions.argmax(axis=1)
    ground = ground.argmax(axis=1)
    accuracy = np.mean(predictions == ground)

    log("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (float(accuracy), microPrecision, microRecall, microF1))
    return accuracy, microPrecision, microRecall, microF1


def writeNormalisedData(dataFilePath, texts):
    """
    Write normalised data to a file
    Input:
        dataFilePath : Path to original train/test file that has been processed
        texts : List containing the normalised 3 turn conversations, separated by the <eos> tag.
    """
    normalisedDataFilePath = dataFilePath.replace(".txt", "_normalised.txt")
    with io.open(normalisedDataFilePath, 'w', encoding='utf8') as fout:
        with io.open(dataFilePath, encoding='utf8') as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                line = line.strip().split('\t')
                normalisedLine = texts[lineNum].strip().split('<eos>')
                fout.write(line[0] + '\t')
                # Write the original turn, followed by the normalised version of the same turn
                fout.write(line[1] + '\t' + normalisedLine[0] + '\t')
                fout.write(line[2] + '\t' + normalisedLine[1] + '\t')
                fout.write(line[3] + '\t' + normalisedLine[2] + '\t')
                try:
                    # If label information available (train time)
                    fout.write(line[4] + '\n')
                except:
                    # If label information not available (test time)
                    fout.write('\n')


def getEmbeddingMatrix(wordIndex):
    """
    Populate an embedding matrix using a word-index. If the word "happy" has an index 19,
       the 19th row in the embedding matrix should contain the embedding vector for the word "happy".
    Input:
        wordIndex : A dictionary of (word : index) pairs, extracted using a tokeniser
    Output:
        embeddingMatrix : A matrix where every row has 100 dimensional GloVe embedding
    """
    embeddingsIndex = {}
    # Load the embedding vectors from ther GloVe file
    with io.open(os.path.join(GLOVE_DIR, GLOVE_FILE), encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector

    print('Found %s word vectors.' % len(embeddingsIndex))

    # Minimum word index of any word is 1. 
    embeddingMatrix = np.zeros((len(wordIndex) + 1, EMBEDDING_DIM))
    for word, i in wordIndex.items():
        embeddingVector = embeddingsIndex.get(word)
        if embeddingVector is not None:
            # words not found in embedding index will be all-zeros.
            embeddingMatrix[i] = embeddingVector

    return embeddingMatrix


def predictAndSave(model, testSequences, tokenizer=None, load=False):
    if load and tokenizer:
        model = tf.keras.models.load_model(os.path.join(sys.path[0], SAVED_MODELS_DIR, (MODEL_NAME + '.h5') % (NUM_EPOCHS, int(LEARNING_RATE * (10 ** 5)), LSTM_DIM, BATCH_SIZE)))
        testTexts = preprocessData(TEST_DATA_PATH, mode="test")[1]
        testSequences = tokenizer.texts_to_sequences(testTexts)

    print("Creating solutions file...")
    testData = tf.keras.preprocessing.sequence.pad_sequences(testSequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(testData, batch_size=BATCH_SIZE)
    predictions = predictions.argmax(axis=1)

    with io.open(os.path.join(SOLUTION_DIR, (MODEL_NAME + '-solution.txt') % (NUM_EPOCHS, int(LEARNING_RATE * (10 ** 5)), LSTM_DIM, BATCH_SIZE)), "w", encoding="utf8") as fout:
        fout.write('\t'.join(["id", "turn1", "turn2", "turn3", "label"]) + '\n')
        with io.open(TEST_DATA_PATH, encoding="utf8") as fin:
            fin.readline()
            for lineNum, line in enumerate(fin):
                fout.write('\t'.join(line.strip().split('\t')[:4]) + '\t')
                fout.write(label2emotion[predictions[lineNum]] + '\n')


def main():
    with open(os.path.join(CONFIG_DIR, CONFIG_FILE)) as configfile:
        config = json.load(configfile)

    global TRAIN_DATA_PATH, TEST_DATA_PATH, SOLUTION_DIR, GLOVE_DIR
    global NUM_FOLDS, NUM_CLASSES, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
    global BATCH_SIZE, LSTM_DIM, DROPOUT, NUM_EPOCHS, LEARNING_RATE

    TRAIN_DATA_PATH = config["train_data_path"]
    TEST_DATA_PATH = config["test_data_path"]
    SOLUTION_DIR = config["solution_dir"]
    GLOVE_DIR = config["glove_dir"]

    NUM_FOLDS = config["num_folds"]
    NUM_CLASSES = config["num_classes"]
    MAX_NB_WORDS = config["max_nb_words"]
    MAX_SEQUENCE_LENGTH = config["max_sequence_length"]
    EMBEDDING_DIM = config["embedding_dim"]
    BATCH_SIZE = config["batch_size"]
    LSTM_DIM = config["lstm_dim"]
    DROPOUT = config["dropout"]
    LEARNING_RATE = config["learning_rate"]
    NUM_EPOCHS = config["num_epochs"]

    print("Processing training data...")
    trainIndices, trainTexts, labels = preprocessData(TRAIN_DATA_PATH, mode="train")
    writeNormalisedData(TRAIN_DATA_PATH, trainTexts)

    print("Processing test data...")
    testIndices, testTexts = preprocessData(TEST_DATA_PATH, mode="test")
    writeNormalisedData(TEST_DATA_PATH, testTexts)

    print("Extracting tokens...")
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(trainTexts)
    trainSequences = tokenizer.texts_to_sequences(trainTexts)
    testSequences = tokenizer.texts_to_sequences(testTexts)

    wordIndex = tokenizer.word_index
    print("Found %s unique tokens." % len(wordIndex))

    print("Populating embedding matrix...")
    embeddingMatrix = getEmbeddingMatrix(wordIndex)

    data = tf.keras.preprocessing.sequence.pad_sequences(trainSequences, maxlen=MAX_SEQUENCE_LENGTH)
    labels = tf.keras.utils.to_categorical(np.asarray(labels))
    print("Shape of training data tensor: ", data.shape)
    print("Shape of label tensor: ", labels.shape)

    # Randomize data
    np.random.shuffle(trainIndices)
    data = data[trainIndices]
    labels = labels[trainIndices]

    # Perform k-fold cross validation
    metrics = {"accuracy": [],
               "microPrecision": [],
               "microRecall": [],
               "microF1": []}

    print("Starting k-fold cross validation...")
    for k in range(NUM_FOLDS):
        log('-' * 40)
        log("Fold %d/%d" % (k + 1, NUM_FOLDS))
        validationSize = int(len(data) / NUM_FOLDS)
        index1 = validationSize * k
        index2 = validationSize * (k + 1)

        xTrain = np.vstack((data[:index1], data[index2:]))
        yTrain = np.vstack((labels[:index1], labels[index2:]))
        xVal = data[index1:index2]
        yVal = labels[index1:index2]
        print("Building model...")
        model = buildModel(embeddingMatrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_DIM, DROPOUT, NUM_CLASSES, LEARNING_RATE)
        model.fit(xTrain, yTrain,
                  validation_data=(xVal, yVal),
                  epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

        predictions = model.predict(xVal, batch_size=BATCH_SIZE)
        accuracy, microPrecision, microRecall, microF1 = getMetrics(predictions, yVal)
        metrics["accuracy"].append(accuracy)
        metrics["microPrecision"].append(microPrecision)
        metrics["microRecall"].append(microRecall)
        metrics["microF1"].append(microF1)

    log("\n============= Metrics =================")
    log("Average Cross-Validation Accuracy : %.4f" % (sum(metrics["accuracy"]) / len(metrics["accuracy"])))
    log("Average Cross-Validation Micro Precision : %.4f" % (sum(metrics["microPrecision"]) / len(metrics["microPrecision"])))
    log("Average Cross-Validation Micro Recall : %.4f" % (sum(metrics["microRecall"]) / len(metrics["microRecall"])))
    log("Average Cross-Validation Micro F1 : %.4f" % (sum(metrics["microF1"]) / len(metrics["microF1"])))
    log("=======================================\n")

    print("Retraining model on entire data to create solutions file")
    model = buildModel(embeddingMatrix, EMBEDDING_DIM, MAX_SEQUENCE_LENGTH, LSTM_DIM, DROPOUT, NUM_CLASSES, LEARNING_RATE)
    model.fit(data, labels, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    model.save(os.path.join(sys.path[0], SAVED_MODELS_DIR, (MODEL_NAME + '.h5') % (NUM_EPOCHS, int(LEARNING_RATE * (10 ** 5)), LSTM_DIM, BATCH_SIZE)))

    predictAndSave(model, testSequences)

    log("Completed. Model parameters: ")
    log("Learning rate : %.3f, LSTM Dim : %d, Dropout : %.3f, Batch_size : %d" % (LEARNING_RATE, LSTM_DIM, DROPOUT, BATCH_SIZE), close=True)


if __name__ == '__main__':
    main()
