import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import seaborn as sns
import logging

# Настраиваем логирование
logging.getLogger("tensorflow").setLevel(logging.ERROR)


# Создает DF с ссылкой на изображение и его тэгом
def data_frame(df, x_col, y_col):
    img_link = []
    img_determinant = []
    for determinant in os.listdir(df):
        determinant_folder = os.path.join(df, determinant)
        for img in os.listdir(determinant_folder):
            img_link.append(os.path.join(determinant_folder, img))
            img_determinant.append(determinant)

    Fseries = pd.Series(img_link, name=x_col)
    Dseries = pd.Series(img_determinant, name=y_col)

    return pd.concat([Fseries, Dseries], axis=1)


# функция для разделения данных на тренировочный, тестовый и валидационные
def test_train_val(data, state):
    train_df, test_df = train_test_split(data, train_size=.9, shuffle=True, random_state=state)
    valid_df, test_df = train_test_split(test_df, train_size=.5, shuffle=True, random_state=state)
    return train_df, test_df, valid_df


# Задает паарметры входного изображения
def img_par(height: int, width: int, channels: int):
    img_shape = (height, width, channels)
    img_size = (height, width)
    length = len(test_df)
    test_steps = int(length / test_batch_size)
    return img_shape, img_size, length, test_steps


# Подгоняет изображения по размерам)
def ImageAugmentation(data, shuffle: bool, x_col: str, y_col: str, img_size: tuple):
    gen = ImageDataGenerator()
    return gen.flow_from_dataframe(data, directory=None, x_col=x_col, y_col=y_col, target_size=img_size,
                                   class_mode='categorical', color_mode='rgb', shuffle=shuffle, batch_size=32)


# Подготавливаем слои модели
def EfficientNetB2_model(img_shape: tuple, lr1: float, lr2: float, dropout_rate: float, class_count: int):
    # Генерируем модель EfficientNetB2
    base_model = tf.keras.applications.EfficientNetB2(include_top=False, weights="imagenet",
                                                      input_shape=img_shape, pooling="max")
    # Выход базовой модели
    x = base_model.output
    # Нормализация данных
    x = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    # Полносвязный слой с регуляризацией
    x = Dense(256, kernel_regularizer=regularizers.l2(lr2), activity_regularizer=regularizers.l1(lr1),
              bias_regularizer=regularizers.l1(lr1), activation="relu")(x)
    # Dropout для регуляризации
    x = Dropout(rate=dropout_rate, seed=None)(x)

    # Выходной слой классификации
    output = Dense(class_count, activation='softmax')(x)
    # Объединяем слои для получения полноценного выходного слоя
    model = Model(inputs=base_model.input, outputs=output)
    # Компилируем модель
    if class_count == 2:
        model.compile(optimizer=Adamax(learning_rate=lr1), loss="binary_crossentropy",
                      metrics=["accuracy"])
    else:
        model.compile(optimizer=Adamax(learning_rate=lr1), loss="categorical_crossentropy",
                      metrics=["accuracy"])
    return model


# Проверка подгонки модели
def prediction(train_data, valid_data, test_data, epochs: int, patience: int, factor: int, batches: int,
               test_steps: int):
    # Прогрев GPU
    warm_up = tf.random.normal([10, 224, 224, 3])
    tf.matmul(warm_up, warm_up, transpose_b=True)
    # Создаем колбэки и подгоняем модель
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=factor, patience=factor)
    ]
    history = model.fit(x=train_data, epochs=epochs, verbose=0,
                        callbacks=callbacks, validation_data=valid_data,
                        validation_steps=None, shuffle=False, initial_epoch=0)

    # Проверяем точность модели на тестовой выборке и выводим результат тестирования
    acc = model.evaluate(test_data, batch_size=test_batch_size, verbose=1,
                         steps=test_steps, return_dict=False)[1] * 100
    msg = f'accuracy on the test set is {acc:5.2f} %'
    print(msg)
    axx = model.evaluate(valid_gen, batch_size=test_batch_size, verbose=1,
                         steps=test_steps, return_dict=False)[1] * 100
    print(f'accuracy on the test set is {axx:5.2f} %')
    return history


# Выводит графики
def charts(history):
    sns.set_style('darkgrid')
    # График точности
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # График потерь
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    print("Train classes:", np.unique(train_gen.classes, return_counts=True))
    print("Val classes:", np.unique(valid_gen.classes, return_counts=True))


# Размер изображений
height = 260
width = 260
channels = 3
# Размер пачки для обучения
batch_size = 32
# Размер пачки для валидации
test_batch_size = 50
# Название стлобцом
x_col = "img-link"
y_col = "determinant"

# Определяем БД
animals_df = "PetImages"
df2 = data_frame(animals_df, x_col, y_col)

# Создаем тренировочную, тестовую, валидационную пачку
train_df, test_df, valid_df = test_train_val(data=df2, state=None)

# Подгтотавливаем нужные данные
img_shape, img_size, length, test_steps = img_par(height, width, channels)

# Аугментируем все три пачки данных
train_gen = ImageAugmentation(train_df, shuffle=True, x_col=x_col, y_col=y_col,
                              img_size=img_size)

test_gen = ImageAugmentation(test_df, shuffle=False, x_col=x_col, y_col=y_col,
                             img_size=img_size)

valid_gen = ImageAugmentation(valid_df, shuffle=True, x_col=x_col, y_col=y_col,
                              img_size=img_size)

# Получаем тэги классов, их кол-во и тренировочные шаги
classes = list(train_gen.class_indices.keys())
class_count = len(classes)
train_steps = np.ceil(len(train_gen.labels) / batch_size)

# Создаем и обучаем модель
model = EfficientNetB2_model(img_shape, lr1=.001, lr2=.05, dropout_rate=.7, class_count=class_count)

history = prediction(train_gen, valid_gen, test_gen, epochs=20,
                     patience=1, factor=.5, batches=train_steps, test_steps=test_steps)

# Выводим график обучения
charts(history)
