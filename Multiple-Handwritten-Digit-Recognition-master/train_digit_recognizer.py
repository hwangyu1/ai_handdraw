import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from emnist import extract_training_samples, extract_test_samples
import math

# EMNIST 데이터셋 로드
train_images, train_labels = extract_training_samples('letters')
test_images, test_labels = extract_test_samples('letters')
train_images_digits, train_labels_digits = extract_training_samples('digits')
test_images_digits, test_labels_digits = extract_test_samples('digits')

# 데이터 전처리
train_images = train_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
test_images = test_images.reshape((-1, 28, 28, 1)).astype('float32') / 255.0

#데이터 크기 및 모양 확인
print("Train images shape:", train_images.shape)
print("Test images shape:", test_images.shape)

# 레이블을 범주형 형태로 변환
num_classes = 62     #숫자10 소문자대문자26
train_labels = to_categorical(train_labels - 1, num_classes)  # 레이블을 0부터 시작하도록 조정
test_labels = to_categorical(test_labels - 1, num_classes)

def show_sample(images, labels, sample_count=10):

  grid_count = math.ceil(math.ceil(math.sqrt(sample_count)))
  grid_count = min(grid_count, len(images), len(labels))

  plt.figure(figsize=(2*grid_count, 2*grid_count))
  for i in range(sample_count):
    plt.subplot(grid_count, grid_count, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images[i], cmap=plt.cm.gray)
    plt.xlabel(labels[i])
  plt.show()
  
model = Sequential()
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 요약 확인
model.summary()

#훈련 손실 (training loss): 0.9149
#훈련 정확도 (training accuracy): 0.7256
#검증 손실 (validation loss): 0.5094
#검증 정확도 (validation accuracy): 0.8697
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)


model.summary()

# 학습률 출력
learning_rate = model.optimizer.lr.numpy()
print(f"Learning Rate: {learning_rate}")

# 검증 세트 생성 및 평가
validation_loss, validation_accuracy = model.evaluate(test_images, test_labels)
print(f"Validation Loss: {validation_loss}, Validation Accuracy: {validation_accuracy}")


# HDF5 파일 형식으로 저장
model.save("my_model.h5")


loaded_model = load_model("my_model.h5")
