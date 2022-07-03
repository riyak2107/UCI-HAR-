import pandas as pd

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

import tensorflow as tf

tf.random.set_seed(4445444)
train_df = pd.read_csv("C:/Users/Riya/Desktop/chi/train.csv")
test_df = pd.read_csv("C:/Users/Riya/Desktop/chi/test.csv")

#train_df.info()

#print(train_df.isna().sum().sum())

#test_df.info()

#print(test_df.isna().sum().sum())

y_train = train_df['Activity'].copy()
X_train = train_df.drop('Activity', axis=1).copy()

y_test = test_df['Activity'].copy()
X_test = test_df.drop('Activity', axis=1).copy()

# print(y_train.value_counts())
num_classes = 6

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

y_train = label_encoder.transform(y_train)
y_test = label_encoder.transform(y_test)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

inputs = tf.keras.Input(shape=(X_train.shape[1],))
x = tf.keras.layers.Dense(64, activation='relu')(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

batch_size = 32
epochs = 25

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train,y_train,validation_split=0.2,batch_size=batch_size,epochs=epochs,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint('./model.h5', save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=1)
       ]
    )

print(model.evaluate(X_test, y_test))
# print("Accuracy : ",model.evaluate(y_test))


'''
Output 
Epoch 1/25
184/184 [==============================] - 1s 5ms/step - loss: 0.6150 - accuracy: 0.7900 - val_loss: 0.2937 - val_accuracy: 0.9218 - lr: 0.0010
Epoch 2/25
184/184 [==============================] - 1s 5ms/step - loss: 0.2392 - accuracy: 0.9104 - val_loss: 0.1926 - val_accuracy: 0.9205 - lr: 0.0010
Epoch 3/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1636 - accuracy: 0.9383 - val_loss: 0.2388 - val_accuracy: 0.9137 - lr: 0.0010
Epoch 4/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1160 - accuracy: 0.9631 - val_loss: 0.1599 - val_accuracy: 0.9388 - lr: 1.0000e-04
Epoch 5/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1104 - accuracy: 0.9638 - val_loss: 0.1496 - val_accuracy: 0.9388 - lr: 1.0000e-04
Epoch 6/25
184/184 [==============================] - 1s 5ms/step - loss: 0.1067 - accuracy: 0.9640 - val_loss: 0.1508 - val_accuracy: 0.9381 - lr: 1.0000e-04
Epoch 7/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1035 - accuracy: 0.9670 - val_loss: 0.1482 - val_accuracy: 0.9375 - lr: 1.0000e-05
Epoch 8/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1027 - accuracy: 0.9677 - val_loss: 0.1484 - val_accuracy: 0.9381 - lr: 1.0000e-05
Epoch 9/25
184/184 [==============================] - 1s 3ms/step - loss: 0.1022 - accuracy: 0.9668 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-06
Epoch 10/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9672 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-07
Epoch 11/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-08
Epoch 12/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-09
Epoch 13/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-10
Epoch 14/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-11
Epoch 15/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-12
Epoch 16/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-13
Epoch 17/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-14
Epoch 18/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-15
Epoch 19/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-16
Epoch 20/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-17
Epoch 21/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-18
Epoch 22/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-19
Epoch 23/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-20
Epoch 24/25
184/184 [==============================] - 1s 4ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-21
Epoch 25/25
184/184 [==============================] - 1s 3ms/step - loss: 0.1021 - accuracy: 0.9670 - val_loss: 0.1487 - val_accuracy: 0.9375 - lr: 1.0000e-22
93/93 [==============================] - 0s 2ms/step - loss: 0.1937 - accuracy: 0.9257
[0.19370079040527344, 0.9256871342658997]

Process finished with exit code 0
'''