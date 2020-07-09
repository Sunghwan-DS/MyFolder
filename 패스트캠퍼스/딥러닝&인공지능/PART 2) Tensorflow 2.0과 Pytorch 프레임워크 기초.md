# 1. Tensorflow 2.0과 Pytorch 소개

## 1.1. 챕터 소개

- 각 프레임워크 별 간단 Tutorial
  - 시각화
  - Framework 기초 사용법
  - 예제 데이터 불러오고 다루기
  - Layer 구현 및 파악 & 모델 설계
  - 인공지능 학습



## 1.1.1. TensorFlow

- 1.x에 비해 정말 쉬워졌음
- Numpy Array와 호환이 쉬움
- TensorBoard, TFLite, TPU
- 여전히 많은 사용자들이 사용



### 1.1.2. PyTorch

- Dynamic Graph & Define by Run
- 쉽고 빠르며 파이써닉하다
- 어마어마한 성장률



# 2. tensorflow 2.0 기초 사용법

```python
import numpy as np
import tensorflow as tf
```



## 2.1. Tensor 생성

- []
  
  - List 생성
  
  ```python
  [1, 2, 3]
  [[1, 2, 3], [4, 5, 6]]
  ```



### 2.1.1. Array 생성

tuple이나 list 둘 다 np.array()로 씌어서 array를 만들 수 있다.

```python
arr = np.array([1, 2, 3])
print(arr.shape)
arr = np.array([[1, 2, 3], [1, 2, 3]])
print(arr.shape)
```

```
(3, )
(2, 3)
```



### 2.1.2. Tensor 생성

tf.constant()

- list -> Tensor

  ```python
  tf.constant([1, 2, 3])
  ```

- tuple -> Tensor

  ```python
  tf.constant((1, 2, 3), (1, 2, 3))
  ```

- Array -> Tensor

  ```python
  arr = np.array([1, 2, 3])
  tf.constant(arr)
  ```

  

### 2.1.3. Tensor에 담긴 정보 확인

- shape 확인

  ```python
  arr = np.array([1, 2, 3])
  tensor = tf.constant(arr)
  print(tensor.shape)
  ```

  ```
  TensorShape([3])
  ```

- data type 확인

  - 주의: Tensor 생성할 때도 data type을 정해주지 않기 때문에 data type에 대한 혼동이 올 수 있다.
  - Data Type에 따라 모델의 무게나 성능 차이에도 영향을 줄 수 있다.

  ```python
  print(tensor.dtype)
  ```

  ```
  tf.int32
  ```

- data type 정의

  ```python
  tf.constant([1, 2, 3], dtype=tf.int32)
  ```

- data type 변환

  - Numpy에서 astype()을 주었듯이, TensorFlow에서는 tf.cast를 사용한다.

  ```python
  arr = np.array([1, 2, 3], dtype=np.float32)
  
  arr.astype(np.uint8)
  
  tf.cast(tensor, dtype=tf.uint8)
  ```

- Tensor에서 Numpy 불러오기

  - .numpy()

  ```python
  tensor.numpy()
  ```

  - np.array()

  ```python
  np.array(tensor)
  ```

- type()를 사용하여 numpy array로 변환된 것 확인

  ```python
  type(tensor.numpy())
  ```

  ```
  numpy.ndarray
  ```

  

### 2.1.4. 난수 생성

![image-20200708194814626](image/image-20200708194814626.png)

- Normal Distribution은 중심극한 이론에 의한 연속적인 모양

- Uniform Distribution은 중심 극한 이론과는 무관하며 불연속적이며 일정한 분포

- numpy에서는 normal distribution을 기본적으로 생성

  - np.random.randn()

  ```python
  np.random.randn(9)
  ```

  ```
  array([ 2.24804254,  0.46328841,  0.23765128,  0.27295027, -0.56958741,
          2.1216173 ,  0.32800235,  1.14004362,  0.71363775])
  ```

- tf.random.normal

  - TensorFlow에서 Normal Distribution

  ```python
  tf.random.normal([3, 3])
  ```

  ```
  <tf.Tensor: shape=(3, 3), dtype=float32, numpy=
  array([[-1.9345311 ,  0.1974105 ,  1.1650361 ],
         [-0.14203091,  1.8918549 , -0.35417143],
         [-1.1897193 ,  0.6024721 , -0.8902905 ]], dtype=float32)>
  ```

- tf.random.uniform

  - TensorFlow에서 Uniform Distribution

  ```python
  tf.random.uniform([4, 4])
  ```

  ```
  <tf.Tensor: shape=(4, 4), dtype=float32, numpy=
  array([[0.44426167, 0.07362592, 0.5645573 , 0.73015726],
         [0.3714888 , 0.2901349 , 0.8858013 , 0.25791526],
         [0.65857875, 0.9465771 , 0.28069663, 0.6894628 ],
         [0.27670312, 0.23041391, 0.7504574 , 0.8873346 ]], dtype=float32)>
  ```



# 3. 예제 dataset 소개(MNIST) 및 불러오기

## 3.1. Data Preprocess (MNIST)

```python
import numpy as np
import matplotlibpyplot as plt

import tensorflow as tf

%matplotlib inline
```



## 3.2. 데이터 불러오기

TensorFlow에서 제공해주는 데이터셋(MNIST) 예제 불러오기

```python
from tensorflow.keras import datasets
```

- 데이터 shape 확인하기

```python
mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
print(train_x.shape)
```

```
(60000, 28, 28)
```



## 3.3. Image Dataset 들여다보기

불러온 데이터셋에서 이미지 데이터 하나만 뽑아서 시각화까지 확인.

- 데이터 하나만 뽑기

  ```python
  image = train_x[0]
  print(image.shape)
  ```

  ```
  (28, 28) # (28, 28, 3)이 아니므로 gray scale
  ```

- 시각화해서 확인

  ```python
  plt.imshow(image, 'gray')
  plt.show()
  ```

  

## 3.4. Channel 관련

[Batch Size, Height, Width, Channel]

GrayScale이면 1, RGH이면 3으로 만들어줘야 한다.

- 다시 shape로 데이터 확인

  ```python
  print(train_x.shape)
  ```

  ```
  (60000, 28, 28)
  ```

- 데이터 차원수 늘리기 (numpy)

  ```python
  expanded_data = np.expand_dims(train_x, -1)
  print(expanded_data.shape)
  ```

  ```
  (60000, 28, 28, 1)
  ```

- TensorFlow 패키지 불러와 데이터 차원수 늘리기 (tensorflow)

  ```python
  new_train_x = tf.expand_dims(train_x, -1)
  print(new_train_x.shape)
  ```

  ```
  TensorShape([60000, 28, 28, 1])
  ```

- TensorFlow 공식홈페이지에서 가져온 방법 tf.newaxis

  ```python
  train_x[..., tf.newaxis].shape
  ```

  ```
  (6000, 28, 28, 1)
  ```

  ```python
  reshaped = train_x.reshape([60000, 28, 28, 1])
  reshaped.shape
  ```

  ```
  (60000, 28, 28, 1)
  ```



*주의 사항

matplotlib로 이미지 시각화 할 때는 gray scale의 이미지는 3번째 dimension이 없으므로, 2개의 dimension으로 gray scale로 차원 조절해서 넣어줘야 한다.

- new_train_x[0] -> new_train_x[0, :, :, 0]

  ```python
  new_train_x = train_x[..., tf.newaxis]
  new_train_x.shape
  ```

  ```
  (60000, 28, 28, 1)
  ```

  ```python
  disp = new_train_x[0, :, :, 0]
  
  plt.imshow(disp, 'gray')
  plt.show()
  ```

  