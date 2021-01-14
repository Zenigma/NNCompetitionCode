import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import models, layers, losses
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
np.set_printoptions(suppress=True, precision=3, threshold=50)
#tf.random.set_random_seed(3)

def denormalise(arr, min, rang):
    if min is None:
        min = 0.0
    if rang is None:
        rang = 1.0
    return (arr*rang) + min

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

def print_results(best_model, input, output, threshold, annot="train", xmin=None, xrang=None, ymin=None, yrang=None, zmin=None, zrang=None):
    score = best_model.evaluate(input, output, verbose=2)
    results = best_model.predict(input)
    results = denormalise(np.array(results).flatten(), zmin, zrang)
    error = (results - output)**2
    mean_sqr = np.average(error)
    x = denormalise(input[:, 0], xmin, xrang)
    y = denormalise(input[:, 1], ymin, yrang)
    for idx, res in enumerate(results):
        if error[idx] >= threshold:
            print(f"{idx:6d} {annot:10s}  predict z {res:9.4f} real z {output[idx]:9.4f}   x {x[idx]:9.4f}   y {y[idx]:9.4f}   error {error[idx]:9.4f}")    
    print(f"score for {annot:10s} {score[0]:.5f}")

# reads train data and splits it into x,y,z points
# partition into test and train to cross validate
df = pd.read_csv("train.csv")
x = df.iloc[:,0]
y = df.iloc[:,1]
z = df.iloc[:,2]
df_train = df.sample(frac=0.9, random_state=0)
df_test = df.drop(df_train.index)
df_train.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)
x_train = np.array(df_train.iloc[:,0])
y_train = np.array(df_train.iloc[:,1])
z_train = np.array(df_train.iloc[:,2])
x_train_norm, xmin, xrang = normalise(x_train, min=None, max=None)
y_train_norm, ymin, yrang = normalise(y_train, min=None, max=None)
normalise_input = True
if normalise_input:
    input = np.array(packTuple(x_train_norm, y_train_norm))
else:
    input = np.array(packTuple(x_train, y_train))
    xmin = ymin = xrang = yrang = None
# transform z_train between 0 and 1
z_train = np.array(z_train)
z_train_norm, zmin, zrang = normalise(z_train, min=None, max=None)
#input= np.vstack((x_train, y_train)).transpose()
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f"train set input x min  {np.min(input[:, 0]):.4f} max {np.max(input[:, 0]):.4f} y min {np.min(input[:, 1]):.4f} max {np.max(input[:, 1]):.4f}")

# test set
x_test = np.array(df_test.iloc[:,0])
y_test = np.array(df_test.iloc[:,1])
z_test = np.array(df_test.iloc[:,2])
x_test_norm = normalise(x_test, min=xmin, max=1.0, rang=xrang)[0]
y_test_norm = normalise(y_test, min=ymin, max=1.0, rang=yrang)[0]
if normalise_input:
    input_test = np.array(packTuple(x_test_norm, y_test_norm))
else:
    input_test = np.array(packTuple(x_test, y_test))
#input_test = np.vstack((x_test, y_test)).transpose()
z_test_norm = (np.array(z_test) - zmin)/zrang
print(f"test  set input x min  {np.min(input_test[:, 0]):.4f} max {np.max(input_test[:, 0]):.4f} y min {np.min(input_test[:, 1]):.4f} max {np.max(input_test[:, 1]):.4f}")

# plots points
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x, y, z, s=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

#plt.show()
fig.savefig("points3d.png")

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
model.add(layers.Lambda(lambda x: x*zrang/zmin))
#tf.optimizers.Adam(learning_rate=0.05)
# Adam
model.compile(optimizer='Adam', 
                loss=losses.MeanSquaredError(),
                metrics=['mean_squared_error'])
print("=============================================================")
model.summary()
# keep best model
earlystopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto')
outpath = "best_v3.h5"
modelcheckpoint = ModelCheckpoint(outpath, monitor='val_loss', save_best_only=True)
# fit everything
print("=============================================================")
print("FITTING")
history = model.fit(
    input,
    z_train, 
    shuffle=True, 
    validation_data=(input_test, z_test),
    callbacks=[earlystopping, modelcheckpoint],
    verbose=2,
    epochs=1000)
# see what is `happening with validation set
print("---------------------------------------------------")
print("Finished up looking at test set")
best_model = tf.keras.models.load_model(outpath)
threshold = 2.0
print_results(best_model, input_test, z_test, threshold, annot="test", xmin=xmin, xrang=xrang, ymin=ymin, yrang=yrang, zmin=None, zrang=None)
print("---------------------------------------------------")
print("Finished up looking at train set")
print_results(best_model, input, z_train, threshold, annot="train", xmin=xmin, xrang=xrang, ymin=ymin, yrang=yrang, zmin=None, zrang=None)
# for outputting predictions of model
df = pd.read_csv("test.csv", header=None)
x = np.array(df.iloc[:,0])
y = np.array(df.iloc[:,1])
x_norm = normalise(np.array(x), min=xmin, max=None, rang=xrang)[0]
y_norm = normalise(np.array(y), min=ymin, max=None, rang=yrang)[0]
if normalise_input:
    output = np.array(packTuple(x_norm, y_norm))
else:
    output = np.array(packTuple(x, y))
print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print(f" OUTPUTTING TEST output \n {output}")
print(f"x test real {np.min(x):.4f} {np.max(x):.4f} y test {np.min(y):.4f} {np.max(y):.4f}")
print(f"x test norm {np.min(output[:, 0]):.4f} {np.max(output[:, 0]):.4f} y test {np.min(output[:, 1]):.4f} {np.max(output[:, 1]):.4f}")

results = best_model.predict(output)
np.savetxt("test_results_v3.csv", np.c_[results], delimiter=",")
for idx, res in enumerate(np.array(results).flatten()):
    print(f"{idx:7d} test predicted z   {res:10.4f} x   {x[idx]:10.4f} y   {y[idx]:10.4f} ")
print(history.history['mean_squared_error'])


# plots out training accuracy
fig = plt.figure()
plt.plot(history.history['mean_squared_error'], label='train_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc='lower right')
plt.show()
fig.savefig("training_v3.png")

fig = plt.figure()
#plt.plot(history.history['mean_absolute_percentage_error'], label='mean_percentage')
#plt.xlabel('Epoch')
#plt.ylabel('MapE')
#fig.savefig("percentagemean.png")

# plots points of predicted
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x, y, np.array(results).flatten(), s=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z_pred')

#plt.show()
fig.savefig("predicted_points3d.png")
