import matplotlib.pyplot as plt
import numpy as np
from tcn import TCN, tcn_full_summary
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
def transform_dataset(dataset, look_back=1):

    dataX = [dataset[i:(i + look_back)]
             for i in range(len(dataset) - look_back-1)]

    dataY = [dataset[i + look_back]
             for i in range(len(dataset) - look_back-1)]
    return np.array(dataX), np.array(dataY)

data = np.genfromtxt('Project1_stock_0319/6269.csv', delimiter=',', dtype=None)[1:]
prices = np.array([close for date, open, high, low, close in data]).astype(np.float64) 

Min=np.min(prices)
Max=np.max(prices)
prices=[(x-Min)/(Max-Min) for x in prices]

look_back = 5
trainX, trainY = transform_dataset(prices, look_back)

model=Sequential([TCN(input_shape=(look_back, 1),activation='relu',return_sequences=False)])
model.compile(loss='mse',optimizer='adam')

train_history = model.fit(trainX, trainY, epochs=1000, batch_size=32, 
                          shuffle=True, validation_split=0.2)

loss = train_history.history['loss']
val_loss = train_history.history['val_loss']

model.save('stock_DNN_model.h5')
