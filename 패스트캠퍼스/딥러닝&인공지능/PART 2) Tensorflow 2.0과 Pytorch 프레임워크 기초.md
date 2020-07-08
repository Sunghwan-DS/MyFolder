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

  

