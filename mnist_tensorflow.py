import tensorflow as tf

# 1. MNIST 데이터 셋을 가져오기
mnist = tf.keras.datasets.mnist

# MNIST로부터 학습용 데이터와 검증용 테스트 데이터 가져오기
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 데이터 전처리
# 8~255.8 사이의 값을 찾는 픽셀 값들을 0~1.0 사이의 값을 갖도록 반환
x_train, x_test = x_train / 255.0, x_test / 255.0

# 3. 학습 모델 구성 방법 정의
model = tf.keras.models.Sequential([
    # MNOST 이미지들의 해상더 28x28 = 총 784개의 값을 1차원 배열로 만듬
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # 학습을 위한 함수로 relu 사용함
    # 분석 대상(784개)보다 작은 수의 Dense로 구성함
    tf.keras.layers.Dense(512, activation=tf.nn.relu),

    # softmax는 여러 개 중에 1개를 선택하기 이해 최적화된 알고리즘
    # 0-9까지 10개의 숫자 중 1개를 선택해야하기 때문에 10개의 Dense를 사용함
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# 4. 학습 모델 컴파일
# optimizer(최적화 연산) : adam 사용
# 손실함수 : sparse_categorical_crossentropy 사용
# metrics(평가방법) : accuracy = 정확도
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# 5. 데이터 학습 실행, 학습은 총 5번 진행함
model.fit(x_train, y_train, epochs=5)

# 6. 정확성 증가
test_loss, test_acc = model.evaluate(x_test, y_test)
print("정확도:", test_acc)

# 7. 학습 모델 저장
model.save("model/myModel.h5")