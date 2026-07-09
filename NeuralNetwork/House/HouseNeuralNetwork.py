import pandas as pd
data = pd.read_csv("House_Rent_Dataset.csv")
y = data["Rent"]
x= data.drop(['Rent','Posted On', 'Area Locality', 'Floor'], axis=1)
import category_encoders as ce
encoder = ce.LeaveOneOutEncoder(return_df=True)
X=encoder.fit_transform(x,y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.25, random_state=42)
#print(X_train)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
model = Sequential()
model.add(Dense(32, input_shape=(8,), activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

#model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data= (x_test,y_test))
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

#7-7-26
import matplotlib.pyplot as plt
plt.plot(model.history.history['loss'], label='train')
plt.plot(model.history.history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
