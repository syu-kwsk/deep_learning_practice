import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def graph():
    df = pd.read_csv('./log_output.csv')

    epoch = df['epoch'].values
    train = df['train_loss'].values
    val = df['val_loss'].values

    blank = [i for i in range(len(epoch)) if (i+1) % 5 != 0]

    epoch = np.delete(epoch, blank)
    train = np.delete(train, blank)
    val = np.delete(val, blank)

    plt.xlabel('epoch')
    plt.plot(epoch, train, label='train_loss')
    plt.plot(epoch, val, label='val_loss')
    plt.legend()

    plt.savefig('learning_curve.png')
    return 0

if __name__ == '__main__':
    graph()
