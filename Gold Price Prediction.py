import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# CSV 파일에서 데이터 불러오기
data_path = 'C:/Users/data8316-22/Downloads/Gold Price (2013-2023).csv'
df = pd.read_csv(data_path)

# 데이터 구조 확인을 위하여 처음 몇 행 표시
print(df.head())

# 'Date'열을 datatime 형식으로 변환
df['Date'] = pd.to_datetime(df['Date'])

# 날짜별로 데이터를 정렬하여 시간순으로 정렬
df = df.sort_values('Date')

# 'Date'열을 데이터 프레임의 index로 설정
df.set_index('Date', inplace=True)

# 쉼표를 제거하고 가격을 float 형으로 변환하여 가격 열 정비
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# 지정된 기간 동안 금 가격 데이터 구분
plt.figure(figsize=(14, 5))
plt.plot(df['Price'])
plt.title('Gold Price (2013-2023)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# 가격 데이터를 신경망 입력에 대해 0과1 범위로 축적
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Price'].values.reshape(-1, 1))

# 데이터를 학습용과 검증용으로 분할 (80% 학습,20% 검증)
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 1)))  # return_sequences를 False로 변경
model.add(Dropout(0.2))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=32, epochs=50)

# 예측 수행 및 형태 변환
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 예측값을 원래 스케일로 변환
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# RMSE 계산
train_rmse = np.sqrt(np.mean(((train_predict - scaler.inverse_transform(y_train.reshape(-1, 1))) ** 2)))
test_rmse = np.sqrt(np.mean(((test_predict - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2)))
print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')

# 시각화를 위한 데이터 준비
train_plot = np.empty_like(scaled_data)
train_plot[:, :] = np.nan
train_plot[time_step:len(train_predict) + time_step, :] = train_predict

test_plot = np.empty_like(scaled_data)
test_plot[:, :] = np.nan
test_plot[len(train_predict) + (time_step * 2) + 1:len(scaled_data) - 1, :] = test_predict

# 결과 시각화
plt.figure(figsize=(14, 5))
plt.plot(scaler.inverse_transform(scaled_data), label='Original Data')
plt.plot(train_plot, label='Train Predict')
plt.plot(test_plot, label='Test Predict')
plt.title('Gold Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()