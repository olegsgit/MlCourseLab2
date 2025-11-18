import os
import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

# список классов (буквы A–J)
classes = list("ABCDEFGHIJ")

def load_data(data_dir):
    x, y = [], []
    for label, cls in enumerate(classes):
        folder = os.path.join(data_dir, cls)
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert('L')          # конверитруем все картинки в ч/б
                arr = np.array(img, dtype=np.float32) / 255  # нормируем значения пикселей из 0-255 в 0-1
                x.append(arr.flatten())                      # переводим картинку 28x28 в вектор 784
                y.append(label)                              # метка класса(A-J)
            except Exception:
                pass  # битые файлы пропускаем
    return np.array(x), np.array(y)

def one_hot(y, num_classes=10):
    oh = np.zeros((y.shape[0], num_classes), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh

# загружаем оба набора
x_large, y_large = load_data("notMNIST_large")
x_small, y_small = load_data("notMNIST_small")

# делим large на train/valid (например, 5/6 train, 1/6 valid)
sss = StratifiedShuffleSplit(n_splits=1, test_size=1/6, random_state=42)
train_idx, valid_idx = next(sss.split(x_large, y_large))

x_train, y_train = x_large[train_idx], y_large[train_idx]
x_valid, y_valid = x_large[valid_idx], y_large[valid_idx]
x_test,  y_test  = x_small, y_small

# сохраняем всё в один файл
os.makedirs("data/processed", exist_ok=True)
np.savez("data/processed/notmnist.npz",
         x_train=x_train, y_train=one_hot(y_train),
         x_valid=x_valid, y_valid=one_hot(y_valid),
         x_test=x_test,   y_test=one_hot(y_test))

print("Данные сохранены в data/processed/notmnist.npz")
print("train:", x_train.shape, y_train.shape)
print("valid:", x_valid.shape, y_valid.shape)
print("test:",  x_test.shape,  y_test.shape)