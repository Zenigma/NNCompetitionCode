from tensorflow.keras import models, layers, losses
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# pack tuple to be input into NN (,2)
def packTuple(x,y):
    result = []
    for index, item in enumerate(x):
        result.append([item,y[index]])
    return result

# reads train data and splits it into x,y,z points
df = pd.read_csv("train.csv")
x = df.iloc[:,0]
y = df.iloc[:,1]
z = df.iloc[:,2]
input = packTuple(x,y)

# plots points
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.scatter(x,y,z,s=1)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

plt.show()

# simple NN with adam optimizer and MSE loss metric
model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(2,)))
model.add(layers.Dense(10, activation='relu'))
model.add(layers.Dense(1, activation='relu'))
model.compile(optimizer='Adam',
                loss=losses.MeanSquaredError(),
                metrics=['mean_squared_error'])

history = model.fit(np.array(input),np.array(z), epochs=50)

# plots out training accuracy
plt.plot(history.history['mean_squared_error'], label='train_mean_squared_error')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend(loc='lower right')
plt.show()


# for outputting predictions of model
df = pd.read_csv("test.csv", header=None)
x = df.iloc[:,0]
y = df.iloc[:,1]
output = packTuple(x,y)

results = model.predict(np.array(output))
np.savetxt("test_results.csv", np.c_[results], delimiter=",")
