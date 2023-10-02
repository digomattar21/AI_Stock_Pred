#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import yfinance as yf
scaler = MinMaxScaler(feature_range=(0, 1))

#Asking for user to input the ticker he wants to predict
inpt = input('Input Ticker: ')
epoch = int(input('Input # of EPOCHS wished: '))

#Grabbing ticker from yahoo finance
ticker = yf.Ticker(inpt)

#Getting historical data for more precision and converting to a readable csv file
hist = ticker.history(period='max')
data = pd.DataFrame(hist)
data.to_csv('stock_df.csv')
df = pd.read_csv('stock_df.csv')

# printing legnth to see how many lines we are working with
print('Length:', len(df))

#setting date as index
df['Date'] = pd.to_datetime(df.Date)
df.index = df['Date']

#creating dataframe with date and the target variable
data = df.sort_index(ascending=True, axis=0)
novo = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])


#Splitting the DF, we cannot use the train_test_split since it will destroy our time stamp
data = df.sort_index(ascending=True, axis=0)
novo = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])

for i in range(0,len(data)):
    novo['Date'][i] = data['Date'][i]
    novo['Close'][i] = data['Close'][i]

#setting index
novo.index = novo.Date
novo.drop('Date', axis=1, inplace=True)

#creating train and test sets
df_novo = novo.values

# splitting into train and validation (divide by 1.42 to get approx 70% of dataset into training)

split = int(len(df_novo)/1.42)
print('Split #:',split)

train = df_novo[0:split,:]
valid = df_novo[split:,:]

#getting dataset to format LSTM will understand
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_novo)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Using LSTM, fitting model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=epoch, batch_size=1, verbose=2)

#predicting values using 60 days bc of covid
inputs = novo[len(novo) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

#Printing RMS to see accuracy
rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
print('RMS:',rms)

#Plotting the results for comparison
train_result = novo[:split]
validation = novo[split:]

validation['Prediction']=0
validation['Predictions']=closing_price
plt.figure(figsize=(20,20))
plt.plot(train_result['Close'])
plt.plot(validation[["Close","Predictions"]])
plt.legend(loc='upper left')
plt.show()


# %%
