# Fruits-360: классификация фруктов и овощей с помощью CNN

Лабораторная работа № 9 по дисциплине «Нейронные сети».
Выполнил: Бронников Д.С., группа МОм-25-1.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Danil20340/fruits-360/blob/main/CNN%20Fruits-360.ipynb)

Свёрточная нейронная сеть (4 свёрточных блока, ≈ 5,2 млн параметров) обучается на датасете
[Fruits-360](https://github.com/Horea94/Fruit-Images-Dataset) — 131 класс фруктов и овощей,
изображения 100×100 RGB — и по изображению определяет вид плода.

## Файлы

| Файл | Что это |
|---|---|
| [`CNN Fruits-360.ipynb`](CNN%20Fruits-360.ipynb) | отчёт: подготовка данных, модель, обучение с замером времени, оценка качества, сохранение и загрузка модели |
| `fruits360_cnn_model.h5` | обученная модель (Keras, HDF5, без состояния оптимизатора) |
| `class_names.txt` | названия 131 класса в порядке выходов модели |

## Данные

Датасет в репозиторий не копируется: архив весит около 760 МБ, это больше лимита GitHub на один файл.
Ноутбук скачивает его по прямой ссылке
https://github.com/Horea94/Fruit-Images-Dataset/archive/refs/heads/master.zip

## Быстрый инференс

Нормализация встроена в модель (слой `Rescaling`), поэтому на вход подаётся обычное изображение
100×100 RGB со значениями 0..255.

```python
import urllib.request

import numpy as np
import tensorflow as tf

base = "https://github.com/Danil20340/fruits-360/raw/main"
urllib.request.urlretrieve(f"{base}/fruits360_cnn_model.h5", "fruits360_cnn_model.h5")
urllib.request.urlretrieve(f"{base}/class_names.txt", "class_names.txt")

model = tf.keras.models.load_model("fruits360_cnn_model.h5", compile=False)
with open("class_names.txt", encoding="utf-8") as f:
    class_names = [line.strip() for line in f if line.strip()]

img = tf.keras.utils.load_img("apple.jpg", target_size=(100, 100))
probs = model.predict(tf.expand_dims(tf.keras.utils.img_to_array(img), 0), verbose=0)[0]
print(class_names[np.argmax(probs)], f"{np.max(probs):.2%}")
```
