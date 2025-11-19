import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Загружаем готовый файл
data = np.load("data/processed/notmnist.npz")

x_train, y_train = data["x_train"], data["y_train"]
x_valid, y_valid = data["x_valid"], data["y_valid"]
x_test,  y_test  = data["x_test"],  data["y_test"]

print("train:", x_train.shape, y_train.shape)
print("valid:", x_valid.shape, y_valid.shape)
print("test:",  x_test.shape,  y_test.shape)

model = models.Sequential([
    layers.Input(shape=(784,)),                 # вход: 784 пикселя
    layers.Dense(512, activation='sigmoid'),       # скрытый слой 1: 512 нейронов и relu активация(кусочно-линейная)   
    layers.Dense(256, activation='sigmoid'),       # скрытый слой 2: 256 нейронов
    layers.Dense(128, activation='sigmoid'),       # скрытый слой 3: 128 нейронов
    layers.Dense(10, activation='softmax')      # выход: 10 классов
])

opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    x_train, y_train,
    validation_data=(x_valid, y_valid),
    epochs=100,            
    batch_size=256,
    verbose=2
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print("Точность на тесте:", test_acc)

# Сохраняем модель в формате HDF5 (.h5)
model.save("models/notmnist_mlp4.h5")