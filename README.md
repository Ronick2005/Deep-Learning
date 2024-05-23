# basic nn model
## Model Architecture
```python
ai_brain = Sequential()
ai_brain.add(Dense(5, activation = "relu", input_shape = X_train.shape))
ai_brain.add(Dense(10, activation = "relu"))
ai_brain.add(Dense(1))
```
## Compile Model
```python
ai_brain.compile(optimizer = 'sgd', loss = 'mse')
```
## Fit Model
```python
ai_brain.fit(X_train1, y_train, epochs = 100)
```

# nn classification
## Model Architecture
```python
ai_brain = Sequential()
ai_brain.add(Dense(5, activation = "relu", input_shape = (X_train.shape[1],)))
ai_brain.add(Dense(10, activation = "relu"))
ai_brain.add(Dense(10, activation = "relu"))
ai_brain.add(Dense(4, activation = "softmax"))
```
## Compile Model
```python
ai_brain.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
```
## Fit Model
```python
ai_brain.fit(x = X_train_scaled, y = y_train, epochs = 2000, batch_size = 32, validation_data = (X_test_scaled, y_test),)
```

# mnist classification
## Model Architecture
```python
model = keras.Sequential()

model.add(Conv2D(32,(3,3),activation="relu",input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(32,activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(16,activation='relu'))
model.add(Dense(10,activation="softmax"))
```
## Compile Model
```python
model.compile('adam', loss ='categorical_crossentropy', metrics=['accuracy'])
```
## Fit Model
```python
model.fit(X_train_scaled, y_train_onehot, epochs=20, validation_data = (X_test_scaled,y_test_onehot))
```

# malaria cell recognition
## Model Architecture
```python
model=Sequential()
model.add(layers.Input(shape= image_shape))
model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
model.add(layers.MaxPool2D())
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D())
model.add(layers.Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
```
## Compile Model
```python
model.compile(optimizer="adam",loss="binary_crossentropy",metrics="accuracy")
```
## Fit Model
```python
model.fit(train_image_gen,epochs=10,validation_data=test_image_gen)
```

# rnn stock price prediction
## Model Architecture
```python
length = 60
n_features = 1

model = Sequential()

model.add(layers.SimpleRNN(50, input_shape = (length, n_features)))
model.add(layers.Dense(1))
```
## Compile Model
```python
model.compile(optimizer = "adam", loss = "mse")
```
## Fit Model
```python
model.fit(X_train1,y_train,epochs=100, batch_size=32)
```

# named entity recognition
## Model Architecture
```python
input_word = layers.Input(shape=(max_len,))

embedding_layer = layers.Embedding(input_dim=num_words,output_dim=50,
                                   input_length=max_len)(input_word)

dropout = layers.SpatialDropout1D(0.1)(embedding_layer)

bid_lstm = layers.Bidirectional(
    layers.LSTM(units=100,return_sequences=True,
                recurrent_dropout=0.1))(dropout)

output = layers.TimeDistributed(
    layers.Dense(num_tags,activation="softmax"))(bid_lstm)

model = Model(input_word, output)
```
## Compile Model
```python
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
```
## Fit Model
```python
model.fit(x=X_train, y=y_train, validation_data=(X_test,y_test),
          batch_size=50, epochs=3,)
```

# convolutional auto encoder
## Model Architecture
```python
model=Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(16,(5,5),activation='relu'),
    layers.MaxPool2D((2,2),padding='same'),
    layers.Conv2D(8,(3,3),activation='relu'),
    layers.MaxPool2D((2,2),padding='same'),
    layers.Conv2D(8,(3,3),activation="relu",padding='same'),
    layers.UpSampling2D((2,2)),
    layers.Conv2D(16,(5,5),activation='relu',padding='same'),
    layers.UpSampling2D((3,3)),
    layers.Conv2D(1,(3,3),activation='sigmoid')
])
```
## Compile Model
```python
model.compile(optimizer='adam', loss='binary_crossentropy')
```
## Fit Model
```python
model.fit(x_train_noisy,x_train,epochs=5,batch_size=64)
```
