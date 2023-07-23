import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(context='talk', style='whitegrid', palette='deep')

def main():

    directory = "./models/"

    # gmp = os.path.join(directory, "gaussian-mixture-prior", "averaging2", "pi0.5", "sigma21e-6", "test-loss")
    gmp = os.path.join(directory, "gaussian-mixture-prior", "test-loss")
    gp = os.path.join(directory, "gaussian-prior", "300-epochs", "test-loss")
    vsgd = os.path.join(directory, "vanilla-SGD", "with-scheduler", "lr0.05", "test-loss")
    sgd_drop = os.path.join(directory, "SGD-Dropout", "with-scheduler", "test-loss")

    # files = [gmp, gp, vsgd, sgd_drop]
    files = [gmp, vsgd, sgd_drop]

    # names = ['Gaussian-mixture', 'Gaussian', 'Vanilla SGD', "SGD Dropout"]
    names = ['Bayes By Backprop', 'Vanilla SGD', "Dropout"]
    colors = [2, 0, 1]

    fig, ax = plt.subplots(layout='tight', figsize=(10, 6))
    
    palette = sns.color_palette('deep')
    # print(palette)

    for i, file in enumerate(files):
        with open(file, 'rb') as f:
            data = pickle.load(f)
            test_accuracy = np.array(data['test-accuracy'])
            print("Best test for {} error {:.2f}".format(names[i], 100. - np.max(test_accuracy)))
            test_error = 100. - test_accuracy
            ax.plot(test_error, lw=2, label=names[i], color=palette[colors[i]])
    
    ax.set_title('Test error')
    ax.set_xlabel('Epoch')
    ax.set_ylabel("Test Error %")
    ax.set_ylim([0.8, 3])
    ax.legend()

    plt.show()

if __name__ == "__main__":
    main()
