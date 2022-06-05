import os
import io

import numpy as np
import tensorflow as tf

SOLUTION_NAME = '300-glove-emoji-replaced-EP30_LR300e-5_LDim128_BS200.txt'
LOG_DIR = './test'

SOLUTION_DIR = './solutions'
TEST_FILE = './datasets/test.txt'
NUM_CLASSES = 4

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

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
        LOGGER = io.open(os.path.join(LOG_DIR, SOLUTION_NAME), 'w', encoding='utf8')
    LOGGER.write(line)
    if args is not None:
        LOGGER.write(str(args))
    LOGGER.write('\n')

    if close:
        LOGGER.close()
        LOGGER = None


def getMetrics(predictions, ground):
    truePositives = np.sum(predictions * ground, axis=0)
    falsePositives = np.sum(np.clip(predictions - ground, 0, 1), axis=0)
    falseNegatives = np.sum(np.clip(ground - predictions, 0, 1), axis=0)

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

    log("Accuracy : %.4f, Micro Precision : %.4f, Micro Recall : %.4f, Micro F1 : %.4f" % (float(accuracy), microPrecision, microRecall, microF1), close=True)

def main():
    true = []
    predicted = []
    with io.open(TEST_FILE, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line = line.strip().split('\t')
            label = emotion2label[line[4]]
            true.append(label)
    true = tf.keras.utils.to_categorical(np.asarray(true))

    with io.open(os.path.join(SOLUTION_DIR, SOLUTION_NAME), encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line = line.strip().split('\t')
            label = emotion2label[line[4]]
            predicted.append(label)
    predicted = tf.keras.utils.to_categorical(np.asarray(predicted))
    getMetrics(predicted, true)


if __name__ == '__main__':
    main()