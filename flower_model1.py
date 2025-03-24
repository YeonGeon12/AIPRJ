from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib

from tensorflow.python.layers.normalization import normalization

# 학습시킬 이미지 파일 경로
data_dir = pathlib.Path("images")
print("data_dir", data_dir)

# 학습데이터 수 출력하기
image_count = len(list(data_dir.glob("*/*.jpg")))
print("image_count", image_count)

######################################################################################
# 1. 데이터 새트 만들기
#######################################################################################

# 로더에 대한 몇가지 매개 변수를 정의
batch_size = 128 # 한번에 학습할 갯수
img_height = 180 # 이미지 높이 크기
img_width = 180 # 이미지 넓이 크기

######################################################################################
# 2. 학습 모델 만들기
# 일반적으로 학습 모델을 만들때 학습데이터의 80%는 학습용, 20%는 검증용으로 사용함
#######################################################################################

# 학습용 데이터셋 생성
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 검증용 데이터셋 생성
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# 각 꽃들의 폴더 이름을 객체 이름으로 사용
class_names = train_ds.class_names
print("class_names", class_names)

# i
#
for image_batch, label_batch in train_ds:
    print("image_batch", image_batch.shape)
    print("labels_batch", label_batch.shape)
    break

# 색상값을 8부터 255 까지 저장하기 위해 이미지 값 전처리
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# 학습을 위한 데이터 세트 생성
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, label_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# 최종 결과의 수
num_classes = 5

######################################################################################
# 3. 학습 모델 구성
# 학습을 위한 신경망 구성
#######################################################################################
model = Sequential([
    # 이미지 전처리
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # rely 알고리즘 사용
])