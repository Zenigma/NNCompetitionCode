import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from time import time
np.set_printoptions(suppress=True, precision=3, threshold=50)
#tf.random.set_seed(3)

def denormalise(arr, min, rang):
    if min is None:
        min = 0.0
    if rang is None:
        rang = 1.0
    return (arr*rang) + min

def evaluate_training(outpath, input, input_test, datadic, threshold=2.0, verbose=True):
    """evaluate_training see how this training did
    Args:
        outpath         string      the model to be evaluated
        threshold       float       print out anything that does not meet this criteria
    """
    # see what is `happening with validation set
    if verbose:
        print("---------------------------------------------------")
        print("Finished up looking at test set")
    best_model = tf.keras.models.load_model(outpath)
    score_test, error_test = print_results(best_model, input_test, datadic["ztest"], threshold, annot="test", xmin=datadic["xmin"], xrang=datadic["xrang"], ymin=datadic["ymin"], yrang=datadic["yrang"], zmin=None, zrang=None, verbose=verbose)
    if verbose:
        print("---------------------------------------------------")
        print("Finished up looking at train set")
    score_train, error_train = print_results(best_model, input, datadic["ztrain"], threshold, annot="train", xmin=datadic["xmin"], xrang=datadic["xrang"], ymin=datadic["ymin"], yrang=datadic["yrang"], zmin=None, zrang=None, verbose=verbose)
    # transform weights between 0 and 1
    error_train /= np.max(error_train)
    return score_test, score_train, error_train

def normalise(arr, min=None, max=None, rang=None):
    if min is None:
        min = np.min(arr)
    if max is None:
        max = np.max(arr)
    if rang is None:
        rang = max - min
    normed = (arr - min)/rang
    return normed, min, rang

def normalise2D(input):
    print(input)

# pack tuple to be input into NN (,2)
def packTuple(x,y):
    result = []
    for index, item in enumerate(x):
        result.append([item,y[index]])
    return result

def plot_accuracy(history, outpath="training_v4.png"):
    # plots out training accuracy
    fig = plt.figure()
    plt.plot(history.history['mean_squared_error'], label='train_mean_squared_error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend(loc='lower right')
    fig.savefig(outpath)

def plot_dist(x, y, z, xlabel='x', ylabel='y', zlabel='z', save_fig="points3d.png"):
    """plot_dist plot the 3d distribution """
    # plots points
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter(x, y, z, s=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    #plt.show()
    if not save_fig is None:
        fig.savefig(save_fig)
    return True

def print_results(best_model, input, output, threshold, annot="train", xmin=None, xrang=None, ymin=None, yrang=None, zmin=None, zrang=None, verbose=True):
    score = best_model.evaluate(input, output, verbose=2)
    results = best_model.predict(input)
    results = denormalise(np.array(results).flatten(), zmin, zrang)
    error = (results - output)**2
    mean_sqr = np.average(error)
    x = denormalise(input[:, 0], xmin, xrang)
    y = denormalise(input[:, 1], ymin, yrang)
    if verbose:
        for idx, res in enumerate(results):
            if error[idx] >= threshold:
                print(f"{idx:6d} {annot:10s}  predict z {res:9.4f} real z {output[idx]:9.4f}   x {x[idx]:9.4f}   y {y[idx]:9.4f}   error {error[idx]:9.4f}")    
        print(f"score for {annot:10s} {score[0]:.7f}")
    return score[0], error

def prepare_data(csv_file, train_frac=0.9, normalise_input=True, verbose=True):
    """prepare_data prepare all the data into test and training set

    Args:
        csv_file        string      name_of training csv file
        train_frac      float       the percentage to use in training
        normalise_input bool        normalise the inputdata between 0 and 1
    """
    # reads train data and splits it into x,y,z points
    # partition into test and train to cross validate
    outdic = {"xmin":None, 
                "ymin":None, 
                "xrang":None, 
                "yrang":None, 
                "ztest":None, 
                "ztrain":None,
                "zmin":None,
                "zrang":None
                }
    df = pd.read_csv(csv_file)
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    z = df.iloc[:,2]
    df_train = df.sample(frac=train_frac, random_state=0)
    df_test = df.drop(df_train.index)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    x_train = np.array(df_train.iloc[:,0])
    y_train = np.array(df_train.iloc[:,1])
    z_train = np.array(df_train.iloc[:,2])
    x_train_norm, xmin, xrang = normalise(x_train, min=None, max=None)
    y_train_norm, ymin, yrang = normalise(y_train, min=None, max=None)
    outdic["xmin"] = xmin
    outdic["ymin"] = ymin
    outdic["xrang"] = xrang
    outdic["yrang"] = yrang
    if normalise_input:
        input = np.array(packTuple(x_train_norm, y_train_norm))
    else:
        input = np.array(packTuple(x_train, y_train))
        xmin = ymin = xrang = yrang = None
    # transform z_train between 0 and 1
    z_train = np.array(z_train)
    outdic["ztrain"] = z_train
    z_train_norm, zmin, zrang = normalise(z_train, min=None, max=None)
    outdic["zmin"] = zmin
    outdic["zrang"] = zrang
    #input= np.vstack((x_train, y_train)).transpose()
    if verbose:
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(f"train set input x min  {np.min(input[:, 0]):.4f} max {np.max(input[:, 0]):.4f} y min {np.min(input[:, 1]):.4f} max {np.max(input[:, 1]):.4f}")
    # test set
    x_test = np.array(df_test.iloc[:,0])
    y_test = np.array(df_test.iloc[:,1])
    z_test = np.array(df_test.iloc[:,2])
    outdic["ztest"] = z_test
    x_test_norm = normalise(x_test, min=xmin, max=1.0, rang=xrang)[0]
    y_test_norm = normalise(y_test, min=ymin, max=1.0, rang=yrang)[0]
    if normalise_input:
        input_test = np.array(packTuple(x_test_norm, y_test_norm))
    else:
        input_test = np.array(packTuple(x_test, y_test))
    #input_test = np.vstack((x_test, y_test)).transpose()
    z_test_norm = (np.array(z_test) - zmin)/zrang
    if verbose:
        print(f"test  set input x min  {np.min(input_test[:, 0]):.4f} max {np.max(input_test[:, 0]):.4f} y min {np.min(input_test[:, 1]):.4f} max {np.max(input_test[:, 1]):.4f}")
    # return input and input_test
    return input, input_test, outdic

def prepare_model(datadic, verbose=True):
    """prepare_model to train"""

    # simple NN with adam optimizer and MSE loss metric
    model = models.Sequential()
    model.add(layers.Dense(1000, activation='relu', input_shape=(2,)))
    # could put a lambda in here to normalise
    #model.add(layers.Lambda(lambda x: x[0]*xrang/xmin, x[1]*yrang/ymin))
    #model.add(layers.BatchNormalization())
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1000, activation='relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(10, activation='relu'))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Lambda(lambda x: x*datadic["zrang"]/datadic["zmin"]))
    if verbose:
        print("=============================================================")
        model.summary()
    return model

def model_compile(model, optimizer="Adam", loss_function=losses.MeanSquaredError(), loss_weights=None, metrics=['mean_squared_error']):
    """model_compile compile the model with certain traits
    """
    #tf.optimizers.Adam(learning_rate=0.05)
    # Adam
    model.compile(optimizer=optimizer, 
                loss=loss_function,
                loss_weights=loss_weights,
                metrics=metrics)
    return model

def prepare_output(csv_test, modelpath, plotpath="predicted_points3d_v4.png"):
    """prepare_output prepare the best output
    Args:
        csv_test            string          the csv file output
        modelpath           string          the best model path
    """
    # for outputting predictions of model
    df = pd.read_csv("test.csv", header=None)
    x = np.array(df.iloc[:,0])
    y = np.array(df.iloc[:,1])
    #x_norm = normalise(np.array(x), min=xmin, max=None, rang=xrang)[0]
    #y_norm = normalise(np.array(y), min=ymin, max=None, rang=yrang)[0]
    if normalise_input:
        output = np.array(packTuple(x_norm, y_norm))
    #else:
        output = np.array(packTuple(x, y))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(f" OUTPUTTING TEST output \n {output}")
    print(f"x test real {np.min(x):.4f} {np.max(x):.4f} y test {np.min(y):.4f} {np.max(y):.4f}")
    print(f"x test norm {np.min(output[:, 0]):.4f} {np.max(output[:, 0]):.4f} y test {np.min(output[:, 1]):.4f} {np.max(output[:, 1]):.4f}")
    best_model = tf.keras.models.load_model(modelpath)
    results = best_model.predict(output)
    np.savetxt("test_results_v4.csv", np.c_[results], delimiter=",")
    for idx, res in enumerate(np.array(results).flatten()):
        print(f"{idx:7d} test predicted z   {res:10.4f} x   {x[idx]:10.4f} y   {y[idx]:10.4f} ")
    #print(history.history['mean_squared_error'])
    # print plot of predicted points
    plot_dist(x, y, z, xlabel='x', ylabel='y', zlabel='z_pred', save_fig=plotpath)


def train_model(model, input, input_test, ztrain, ztest, outpath="best_v4.h5", idx=0, verbose=0):
    """train_model train the model """
    # keep best model
    earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto')
    modelcheckpoint = ModelCheckpoint(outpath, monitor='val_loss', save_best_only=True)
    # fit everything
    #print("=============================================================")
    start_time = time()
    print(f"Fitting model for iteration {idx}")
    history = model.fit(
        input,
        ztrain, 
        shuffle=True, 
        validation_data=(input_test, ztest),
        callbacks=[earlystopping, modelcheckpoint],
        verbose=verbose,
        epochs=1000
        )
    time_taken = time() - start_time
    print(f"Fitting model finished for iteration {idx} time taken {time_taken:.2f} seconds {time_taken/60.0:.2f} minutes")
    return history, model

def main():
    """main runs everything"""
    # parameters
    ######################################################################
    normalise_input = True
    niters = 2
    test_file = "test.csv"
    train_file = "train.csv"
    train_frac = 0.9
    error_threshold = 2.0
    # prepare the datasets
    ######################################################################
    best_score = 1000000.0
    best_model = None
    best_history = None
    best_inputs = None
    best_datadic = None
    for idx in range(0, niters):
        print(f"===================== Iteration {idx:6d} Current Best {best_score:10.7f} =========================")
        verbose = idx == 0
        input, input_test, datadic = prepare_data(train_file, train_frac=train_frac, normalise_input=normalise_input, verbose=verbose)
        model = prepare_model(datadic, verbose=verbose)
        model = model_compile(model, optimizer="Adam", loss_function=losses.MeanSquaredError(), loss_weights=None, metrics=['mean_squared_error'])
        outpath = f"best_v4_{idx}.h5"
        history, model = train_model(model, input, input_test, datadic["ztrain"], datadic["ztest"], idx=idx, outpath=outpath, verbose=0)
        # evaluate
        score_test, score_train, error_train = evaluate_training(outpath, input, input_test, datadic, threshold=error_threshold, verbose=verbose)
        print(f"main:: score for iteration {idx:5d} {score_test:10.7f} training score {score_train:10.7f}")
        print(error_train)
        # do it again with weights
        ##################################################################
        #model = model_compile(model, optimizer="Adam", loss_function=losses.MeanSquaredError(), loss_weights=error_train, metrics=['mean_squared_error'])
        #history_2, model = train_model(model, input, input_test, datadic["ztrain"], datadic["ztest"], idx=idx, outpath=outpath, verbose=0)
        #score_test, score_train, error_train = evaluate_training(outpath, input, input_test, datadic, threshold=error_threshold, verbose=verbose)
        #print(f"main:: score for weighted iteration {idx:5d} {score_test:10.7f} training score {score_train:10.7f}")
        # evaluate
        ##################################################################
        if score_test < best_score:
            print(f"main:: best score found iteration {idx:5d} previous best {best_score:10.7f} new best {score_test:10.7f}")
            best_score = score_test
            best_model = outpath
            best_history = history
            best_inputs = (input, input_test)
            best_datadic = datadic
    # write out results for best model
    ######################################################################
    print(f"__________________________________________________________________")
    print(f"main:: writing output for best score {best_score:10.2f}")
    # need to fix datadic for this to work 
    #score = evaluate_training(outpath, best_inputs[0], best_inputs[1], best_datadic, threshold=error_threshold, verbose=True)
    #plot_accuracy(best_history, outpath="training_v4.png")
    #prepare_output(test_file, best_model, plotpath="predicted_points3d_v4.png")
    np.savetxt("test_results_v4.csv", np.c_[results], delimiter=",")

if __name__ == "__main__":
    main() 