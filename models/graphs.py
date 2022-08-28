import matplotlib.pyplot as plt


def graph(x,y, xLabel, yLabel):
    plt.plot(x, y)
    plt.scatter(x, y)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.ylim(0.5, 0.9)
    plt.show()


if __name__ == '__main__':
    x = [10, 30, 50, 75]
    y = [0.8172, 0.8055, 0.8065, 0.8088]
    graph(x,y, 'Number of epochs', 'Validation micro F1')

    x = [50, 100, 150, 200]
    y = [0.8451, 0.8469, 0.8420, 0.8487]
    graph(x,y, 'Batch size', 'Validation micro F1')
