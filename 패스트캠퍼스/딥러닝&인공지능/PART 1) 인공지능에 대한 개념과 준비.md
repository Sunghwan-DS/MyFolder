

# 1. 인공지능 개념이해 - 챕터 소개

## 1.1. 목차

- 코드와 시각적으로 딥러닝 이해
- 깊게 들어가는 이론은 없으니 먼저 쭉 따라 해보시다가 이론 공부 하신 후에 다시 한번 더 보시는 걸 권장
- 일단 기본 CNN과 간단한 프로젝트도 돌릴 수 있을 정도의 수준



## 1.2. 강의 구성

- CNN을 이해하기 위한 전체적인 큰 그림
  - CNN을 이해하기 위해 알아야 할 것
- CNN을 짜기 위한 준비
  - Anaconda 설치부터 여는 방법
  - Numpy부터 Matplotlib 기초



# 2. 인공지능 개념이해 - 전체 구조 및 학습과정

## 2.1. 전체적인 그림

![image-20200625165928068](image/image-20200625165928068.png)

1. 데이터가 들어감
2. 모델이란 것(Resnet 등)이 학습
3. 정답(레이블)과 비교하여 최적화
4. 다시 모델이 적용



## 2.2. Data

- 학습  시키기 위한 데이터. 이 데이터가 모델에 들어감
- 데이터가 생성되고, 데이터에 Transform 변형을 준다거나 모델에 들어가기 전에 데이터 전처리가 들어감
- 이 때 들어갈 때는 Batch로 만들어서 Model에 넣어줌



## 2.3. Model

- LeNet, AlexNet, VGG나 ResNet 등 다양하게 설계된 모델
- Convolution Layer, Pooling 등 다양한 Layer 층들로 구성
- 이 모델 안에 학습 파라미터가 있고, 이 모델이 학습하는 대상



## 2.4. Prediction / Logit

[0.15, 0.3, 0.2, 0.25, 0.1]

- 각 Class별로 예측한 값.
- 여기서 가장 높은 값이 모델이 예상하는 class 또는 정답

[0.0, 0.0, 0.0, 1.0, 0.0]

- 위의 숫자가 정답이라고 할 때 얼마나 틀렸는지 얼마나 맞았는지 확인 가능



## 2.5. Loss / Cost

- 예측한 값과 정답과 비교해서 얼마나 틀렸는지를 확인.
- Cross Entropy 등 다양한 Loss Function들이 있음
- 이 때 계산을 통해 나오는 값이 Loss(Cost, Cost Value 등)이라고 불림
- 이 Loss는 "얼마나 틀렸는지"를 말하며 이 값을 최대한 줄이는 것이 학습의 과정



## 2.6. Optimization

- 앞에서 얻은 Loss 값을 최소화하기 위해 기울기를 받아 최적화된 Variable 값들로 반환
- 이 반환된 값이 적용된 모델은 바로 전에 돌렸을 때의 결과보다 더 나아지게 됨
- 이 때 바로 최적화된 값만큼 바로 움직이는 것이 아닌 Learning Rate만큼 움직인 값이 적용



## 2.7. Result

- 평가 할 때 또는 예측된 결과를 확인 할 때는 예측된 값에서 argmax를 통해 가장 높은 값을 예측한 class라고 둠

[0.15, 0.3, 0.2, 0.25, 0.1]

위의 예측값에서는 0.2가 제일 높은 값이므로 클래스 2가 가장 높다고 봄 (파이썬에선 0으로 시작)

> 0.3이 제일 높지 않나??



# 3. 인공지능 개념이해 - 딥러닝 용어(1)

## 3.1. Model

![image-20200701170535758](image/image-20200701170535758.png)

model을 학습시키기 위한 것.



## 3.2. Layer

![image-20200701170705347](image/image-20200701170705347.png)

여러 층을 쌓았다하여 딥러닝.

> 층이 깊을수록 좋다고는 하나 너무 깊으면 overfitting되기 쉬우며 성능적으로 무겁고 느릴 수 있다.
>
> 깊이 쌓아여 feature를 detail하게 얻을 수 있다.



## 3.3. Convolution (합성곱)

![image-20200701171002618](image/image-20200701171002618.png)

위의 합성곱은 테두리를 잡아주는 필터



## 3.4. Weight / Filter / Kernel / Variable / Bias

- Weight ~ Variable (깊이 들어가면 다른 용어이나 일단 묶어서 생각)
  - 학습하고자 하는 대상
  - Convolution 안의 weight를 학습시킨다.
- Bias
  - y = Wx + b (=> bias)

![image-20200701171248670](image/image-20200701171248670.png)



## 3.5. Pooling Layer

![image-20200701171518474](image/image-20200701171518474.png)

> 뽑아낸 feature들을 Pooling Layer가 압축시켜준다.
>
> 이미지가 가진 가장 큰 특징들을 반으로 줄여준다.



## 3.6. Optimization

![image-20200701171950974](image/image-20200701171950974.png)



## 3.7. Activation Function (활성화 함수)

![image-20200701172100266](image/image-20200701172100266.png)

> 앞에서 뽑은 특징에서 음수와 같은 불필요한 부분을 제거한다. (ReLU)



## 3.8. Softmax

![image-20200701172241279](image/image-20200701172241279.png)

> 가중치
>
> Softmax를 거쳐서 합이 1이 되도록 한다.



# 4. 인공지능 개념이해 - 딥러닝 용어(2)

## 4.1. Cost / Loss / Loss Function

![image-20200702180952385](image/image-20200702180952385.png)

Loss Function : 예측치가 얼마나 틀렸는가를 계산하는 방식



## 4.2. Optimization

![image-20200701171950974](image/image-20200701171950974.png)

> Loss Function으로 계산된 값을 줄이기 위한 행위 (최적화)
>
> 모델을 업데이트 한다.



## 4.3. Learning Rate

![image-20200702183631295](image/image-20200702183631295.png)

> Learning Rate가 너무 낮으면 시간이 오래 걸린다.
>
> Learning Rate가 너무 높으면 minimum point를 찾지 못할수도 있다.



## 4.4. Batch Size

> 모델에 데이터를 넣어주는데 데이터를 한번에 넣어줄 수 없다.
>
> 한번에 몇 개의 데이터를 넣어줄 것인가가 바로 Batch Size. 보통 32, 64, 128...



## 4.5. Epoch / Step

> Epoch : 모델에 같은 데이터를 몇 번 반복하여 학습시킬 것인가.



## 4.6. Train / Validation / Test

![image-20200703092105368](image/image-20200703092105368.png)



## 4.7. Label / Ground Truth

![image-20200703092147530](image/image-20200703092147530.png)

> 정답. 즉, Y에 해당하는 것.



# 5. 인공지능 개념이해 - CNN 모델 구조

## 5.1. Feature Extraction / Classification

![image-20200703092343406](image/image-20200703092343406.png)

> 특징을 추출하는 곳 / 결정을 내리는 곳
>
> 딥러닝이 어떠한 특징들을 뽑아냈는지 이야기하기 어렵다 => 때문에 블랙박스라고 불림



## 5.2. Convolution Layer

![image-20200703092607992](image/image-20200703092607992.png)

> Filter에 따라 가져오는 특징이 다르다.



## 5.3. Pooling Layer (Max Pooling)

![image-20200703092724611](image/image-20200703092724611.png)

> 가장 큰 특징만을 모은다. 압축이라고 이해하면 좋다.



## 5.4. Fully Connected

![image-20200703093703927](image/image-20200703093703927.png)

> 예측하는 부분



## 5.5. Model

> Layer층에 대해 어떤 Layer층이 있을까에 대해 두려워할 필요가 없다.

### 5.5.1. LeNet

![image-20200703095524323](image/image-20200703095524323.png)



### 5.5.2. AlexNet

![image-20200703095636991](image/image-20200703095636991.png)



### 5.5.3. VGG-16

![image-20200703095710612](image/image-20200703095710612.png)



### 5.5.4. ResNet

![image-20200703095742352](image/image-20200703095742352.png)



### 5.5.5. DenseNet

![image-20200703095811320](image/image-20200703095811320.png)





# 6. 인공지능 개발준비 - OS별 Anaconda부터 TensorFlow 및 Pytorch 설치 (Windows)

## 6.1. anaconda 설치

1. 구글에 anaconda download 검색 후, 첫 번째 사이트 접속.

2. 가장 하단에 있는 다운로드 중 파이썬 3.7 버전에 호환되는 다운로드 선택

3. Username이 한국말로 되있는 경우 All users 선택

   ![image-20200704205841994](image/image-20200704205841994.png)

4. 꼭 체크해야 하는 것은 아니지만 path 설정을 해주기 위해 위쪽 체크 선택

   ![image-20200704210019687](image/image-20200704210019687.png)

5. 작업하고자 하는 Workspace를 만들어준 후, `shift+오른쪽 마우스` 를 통해서 PowerShell을 열어 jupyter notebook을 켤 수 있다.



## 6.2. tensorflow 설치

1. (바로 3번으로 넘어가도 됨)https://www.tensorflow.org/ tps://www.tensorflow.org/ 에 접속
2. 설치로 들어가서 설치 코드를 확인
3. power shell을 열어 `$ pip install tensorflow` 를 입력해준다. 만약 pip 버전이 낮다면 `$ pip install --upgrade pip` 를 먼저 입력해준다.
4. anaconda(jupyter notebook)에서 `import tensorflow as tf` 코드를 통해 사용이 가능하다



## 6.3. pytorch 설치

1. https://pytorch.org/ 에 접속

2. 하단에 설치하고자 하는 환경에 대한 설치 코드를 알아낼 수 있다.

   ![image-20200704212726945](image/image-20200704212726945.png)

   ```
   pip install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

3. power shell에 코드를 입력하여 설치
4. `import torch` 코드를 통해서 사용이 가능하다.



# 7. 인공지능 개발준비 - OS별 Anaconda부터 TensorFlow 및 Pytorch 설치 (mac)

생략



# 8. 인공지능 개발준비 - Anaconda 활용 및 단축키

- `Shift+Enter` : 코드블럭 실행 후, 다음 셀로 넘어감.
- `Alt+Enter` : 실행하고 다음셀 생성 후, 새로 생성된 셀로 넘어감.
- `Ctrl+Enter`: 실행한 후, 실행한 셀에 남음.

코드 블럭의 왼편이 파란색일 때는 Command Mode, 초록색일 때는 Edit Mode 이다. Esc를 누르면 Command Mode로, Enter를 누르면 Edit Mode로 전환할 수 있다.



- `dd` : 셀 삭제, 셀 다중선택 후 사용하면 여러 개의 셀을 한번에 삭제 가능.
- `Shift+M` : 셀 병합.
- `Shift+Ctrl+-` : 셀 분할.
- Kernel - Restart : 코드 재실행.
- `a` : 위쪽에 새로운 셀 생성(above).
- `b` : 아래쪽에 새로운 셀 생성(below).
- `m` : 코드블럭 markdown 모드로 변경.

Help - Keyboard Shortcuts 에서 단축키 확인 가능.



# 9. 인공지능 개발준비 - Tensor 이해하기

## 9.1. Tensor

![image-20200704214816812](image/image-20200704214816812.png)

딥러닝에서는 주로 고차원적인 데이터를 많이 사용될 것이기 때문에 Tensor의 개념을 잘 이해하는 것이 중요하다.

Numpy는 그런 고차원적인 데이터를 다루기 쉽게 만들어져 있어 딥러닝을 하게 된다면 늘 접하게 될 것이다.

```python
import numpy as np
```



### 9.1.1. 0차원

- numpy array는 1 또는 5, 10와 같이 숫자 데이터를 arra화 해줄 수 있다.
- Scalar로 들어갔을 때는 shape가 아무것도 없는 것으로 나온다.

```python
arr = np.array(5)
print(arr.shape)
print(arr.ndim)
```

```
()
0
```



### 9.1.2. 1차원

- 숫자가 10과 같이 하나만 들어간다고 해도 [] 리스트를 한번 씌우게 되면 차원이 생긴다.
- 이때는 1차원이 되는건데 numpy 에서 shape를 표현할 때 (1)이 아닌 (1, ) 이런 식으로 표현하게 된다.

```python
arr = np.array([5])
print(arr.shape)
```

```
(1, )
```

- 명심해야 할 것이 있는데 위의 (3, )에서 3은 3이라는 값이 들어간 것이 아닌 shape라는 것이다.
- 1차원에서 3개의 값이 들어갔다는 의미
- 해석하자면 1차원에 3개의 값(value)가 들어가 있는 상태

```python
arr = np.array([1, 2, 3])
print(arr.shape)
```

```
(3, )
```



### 9.1.3. 2차원

- 대괄호를 추가적으로 씌우면 차원이 추가적으로 하나 생김

```python
arr = np.array([[1, 2, 3]])
print(arr.shape)
```

```
(1, 3)
```

- 위의 shape를 다시 보자면 차원이 2개가 있고, 각 차원마다 각각 3개의 값이 들어있다고 생각하면 된다.

```python
arr = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
print(arr.shape)
```

```
(3, 3)
```

- 참고로 0차원 숫자에 대문자에 대괄호를 2번 씌우면 두 개의 차원이 된다.

```python
arr = np.array([[10]])
print(arr.shape)
print(arr.ndim)
```

```
(1, 1)
2
```



### 9.1.4. 다 차원

![image-20200704220540041](image/image-20200704220540041.png)

```python
print(arr.shape)
```

```
(2, 2, 3, 3)
```



![image-20200704220738158](image/image-20200704220738158.png)

```python
print(arr.shape)
```

```
(3, 3, 1)
```



# 10. 인공지능 개발준비 - Numpy 기초(1)

## 10.1. zeros & ones

### 10.1.1. zeros

0으로 채워진 numpy array를 만들 수 있다.

```python
zeros = np.zeros([3, 3])
print(zeros) # anaconda에서 print없이 확인할 때, 아래와 같이 출력되며 print를 이용할 경우에는 array()와 ,가 출력되지 않는다. 그러나 필기 정리를 위해 코드블럭을 여러 개 두는 대신 하나로 합치고 print를 이용해 표기하였다.
```

```
array([[0., 0., 0.],
	   [0., 0., 0.],
	   [0., 0., 0.]])
```



```python
zeros = np.zeros(1)
print(zeros)
```

```
array([0.])
```



### 10.1.2. ones

```python
ones = np.ones([10, 5])
print(ones)
```

```
array([[1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.],
	   [1., 1., 1., 1., 1.]])
```



```python
print(ones * 5)
```

```
array([[5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],
	   [5., 5., 5., 5., 5.],])
```



### 10.1.3. arange

```python
arr = np.arange(5)
print(arr)
```

```
array([0, 1, 2, 3, 4])
```



```python
arr = np.arange(4, 9)
print(arr)
```

```
arr([4, 5, 6, 7, 8])
```



```python
arr = np.arange(9).reshape(3, 3)
print(arr)
```

```
array([[0, 1, 2],
	   [3, 4, 5],
	   [6, 7, 8]])
```



### 10.1.4. Index

```python
nums = [1, 2, 3, 4, 5]
print(nums[2:])

nums = [1, 2, 3, 4, [1, 2, 3]]
print(nums[4])
```

```
[3, 4, 5]
[1, 2, 3]
```



```python
arr = np.arange(9).reshape(3, 3)
print(arr[1][2])
print(arr[1, 2])
```

```
5
5
```



### 10.1.5. Slicing

```python
arr = np.arange(9).reshape(3, 3)
print(arr[1:])
print(arr[1:, 1:])
```

```
array([[3, 4, 5],
	   [6, 7, 8]])
array([[4, 5],
	   [7, 8]])
```



### 10.1.6. Boolean Indexing

```python
data = np.random.randn(3, 3)
print(data)
print(data <= 0) # 전체가 아니라 각각의 value마다 적용
print(data[data <= 0]) # 1차원으로 출력
data[data <= 0] = 1 # 다음의 대입식은 출력이 불가능
print(data)
```

```
array([[-0.65839009, -0.78798502,  0.83520652],
	   [-0.09694742, -0.88494426,  0.11392526],
	   [ 1.01480238,  1.16378774, -0.55390946]])
array([[ True,  True, False],
	   [ True,  True, False],
	   [False, False,  True]])
array([-0.65839009, -0.78798502, -0.09694742, -0.88494426, -0.55390946])
array([[1.        , 1.        , 0.83520652],
	   [1.        , 1.        , 0.11392526],
	   [1.01480238, 1.16378774, 1.        ]])
```



## 10.2. Broadcast

tensorflow나 pytorch로 계산하면 broadcast의 개념도 잘 이해해야 한다.

broadcast는 연산하려는 서로 다른 두 개의 행렬의 shape가 같지 않고, 한쪽의 차원이라도 같거나 또는 값의 갯수가 한 개일 때, 이를 여러 복사를 하여 연산한다.

```python
arr = np.arange(9).reshape(3, 3)
print(arr + 3)
print(arr * 3)
print(arr + np.array([1, 2, 3]))
```

```
array([[ 3,  4,  5],
	   [ 6,  7,  8],
	   [ 9, 10, 11]])
array([[ 0,  3,  6],
	   [ 9, 12, 15],
	   [18, 21, 24]])
array([[ 1,  3,  5],
	   [ 4,  6,  8],
	   [ 7,  9, 11]])
```

> 2차원에 0차원을 더할 때는 0차원의 규모에서 덧셈이 일어나고, 2차원에 1차원을 더할 때는 1차원의 규모에서 덧셈이 일어난다.



## 10.3. Math Function

### 10.3.1. add, multiply

```python
arr = np.arange(9).reshape(3, 3)
print(np.add(arr, 1)) # == arr + 1
print(np.multiply(arr, 3)) # == arr * 3
```

```
arr([[1, 2, 3],
	 [4, 5, 6],
	 [7, 8, 9]])
arr([[ 0,  3,  6],
	 [ 9, 12, 15],
	 [18, 21, 24]])
```



### 10.3.2. sum, max, min, mean(평균값)

```python
arr = np.arange(9).reshape(3, 3)
print(np.sum(arr))
print(np.max(arr))
print(np.min(arr))
```

```
36
8
0
```



특정 차원에 대해서도 계산이 가능하다.

```python
arr = np.arange(27).reshape(3, 3, 3)
print(arr)
print(np.sum(arr, 0))
print(np.sum(arr, 1))
print(np.sum(arr, axis=2)) # 숫자만 적어도 되고 axis=? 형식으로 차원을 정할 수 있다.
print(np.sum(arr, -1)) # 3차원이라 바로 위의 2차원과 답이 같다.
```

```
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],

       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],

       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])
# 0차원
[[27 30 33]
 [36 39 42]
 [45 48 51]]
# 1차원
[[ 9 12 15]
 [36 39 42]
 [63 66 69]]
# 2차원
[[ 3 12 21]
 [30 39 48]
 [57 66 75]]
# -1 == 2차원
[[ 3 12 21]
 [30 39 48]
 [57 66 75]]
```

> 차원 기준에서 함수를 사용할 때는 함수가 적용되는 array의 가장 바깥쪽 껍질부터 차원 수 만큼 들어간다고 생각하면 좋을 것 같다.



### 10.3.3. argmax, argmin

argmax는 array에서 가장 큰 값이 위치해있는 index를 출력하는 함수. 이 때, 차원과 상관없이 앞에서 몇 번째 원소이냐로 나타낸다. 즉, 항상 정수값을 출력한다.

argmin의 경우 가장 작은 원소의 index를 출력한다. 일반적인 max, min 함수와 마찬가지로 같은 값인 경우에는 <strong>먼저 등장하는 원소</strong>의 index를 우선적으로 출력한다.

```python
arr = np.random.randint(10, size=9).reshape(3, 3)
print(arr)
print(np.argmax(arr))
```

```
array([[1, 3, 3],
       [1, 7, 3],
       [4, 6, 6]])
4
```



### 10.3.4. unique

오름차순으로 정렬한 set이라고 생각하면 편할 것 같다. array에서 unique한 값들을 필터링해준다.

```python
arr = np.array([[1, 3, 3], [1, 7, 3], [4, 6, 6]])
print(np.unique(arr))
```

```
array([1, 3, 4, 6, 7])
```



# 11. 인공지능 개발준비 - Numpy 기초(2)

## 11.1. data type

### 11.1.1. dtype

array의 dtype을 본다

```python
arr = np.array([[1, 2, 3], [1, 2, 3]])
print(arr.dtype)
```

```
dtype('int32')
```



### 11.1.2. astype

```python
arr = np.array([[1., 2, 3], [1, 2, 3]])
print(arr.dtype)
arr = arr.astype(np.int32) # .astype 이지만 arr 원본이 수정되지 않는다. 때문에 arr= arr.astype() 형태로 사용해야 한다.
print(arr)
print(arr.dtype)
```

```
dtype('float64')
array([[1, 2, 3],
       [1, 2, 3]])
dtype('int32')
```



array를 선언할 때 dtype를 설정할 수 있다.

```python
arr = np.array([[1., 2, 3], [1, 2, 3]], dtype=np.uint8)
print(arr.dtype)
```

```
dtype('uint8')
```



### 11.1.3. len, ndim

len(arr.shape)를 통해서 차원의 갯수를 확인할 수 있지만, ndim을 통해 차원 수를 return 가능하다.

```python
arr = np.array([[1, 2, 3], [1, 2, 3]])
print(arr.shape)
print(len(arr.shape)) # tuple형 데이터에 len함수를 이용하는 것이기 때문에 ndim과 같은 기능을 할 수 있다.
print(arr.ndim)
```

```
(2, 3) # tuple
2
2
```



## 11.2. reshape

```python
arr = np.array([[1, 2, 3], [1, 2, 3]])
print(arr.shape)
arr = arr.reshape([1, 6])
print(arr.shape)
print(arr.reshape([6]).shape)
```

```
(2, 3)
(1, 6)
(6,)
```



### 11.2.1. reshape, -1 활용

```python
arr = np.array([[1, 2, 3], [1, 2, 3]])
arr = arr.reshape(-1)
print(arr.shape)
arr = arr.reshape(1, -1)
print(arr.shape)
arr = arr.reshape(-1, 3)
print(arr.shape)
```

```
(6,)
(1, 6)
(2, 3)
```

> -1은 남는 양에 대해서 자동으로 채워준다고 생각하면 된다.



### 11.2.2. random array 생성

```python
arr = np.random.randn(8, 8)
print(arr.shape)
arr = arr.reshape([32, 2])
print(arr.shape)
```

```
(8, 8)
(32, 2)
```



### 11.2.3. 3차원으로 늘리기

```python
arr = np.random.randn(8, 8)
arr = arr.reshape(-1, 2, 1)
print(arr.shape)
```

```
(32, 2, 1)
```



## 11.3. Ravel

```python
arr = np.random.randn(8, 8)
print(arr.shape)
arr = arr.ravel() # == arr.reshape(-1)
print(arr.shape)
```

```
(8, 8)
(64,)
```

array의 차원을 1로 바꿔준다. 나중에 배울 layer를 flatten 할 때 같은 기능이라 생각하면 된다.



## 11.4. np.expand_dims()

안의 값은 유지하되 차원 수를 늘리고 싶을 때 사용하는 함수.

```python
arr = np.random.randn(8, 8)
print(arr.shape)
new_arr = np.expand_dims(arr, 0)
print(new_arr.shape)
new_arr = np.expand_dims(arr, -1)
print(new_arr.shape)
```

```
(8, 8)
(1, 8, 8)
(8, 8, 1)
```



# 12. 인공지능 개발준비 - 시각화 기초 (그래프)

## 12.1. Load Packages

```python
import matplotlib.pyplot as plt

%matplotlib inline # jupyter에서만 사용되는 코드로 그래프를 새로운 창으로 띄우지 않도록 한다.
```



## 12.2. Basic Attributes

alpha : 투명도

kind : 그래프 종류. 'line', 'bar', 'barth', 'kde'

logy : Y축에 대해 Log scaling

use_index : 객체의 색인을 눈금 이름으로 사용할지 여부

rot : 눈금 이름 돌리기 (rotating) 0 ~ 360

xticks, yticks : x, y축으로 사용할 값

xlim, ylim : X, Y축의 한계

grid : 축의 그리드를 표현할지 여부



subplots : 각 column에 독립된 subplot 그리기

sharex, sharey : subplots=True 이면 같은 X, Y축을 공유하고 눈금과 한계를 연결

figsize : 생성될 그래프의 크기를 tuple로 지정

title : 그래프의 제목 지정

legend : subplot의 범례 지정

sort_columns : column을 알파벳 순서로 그린다.



## 11.3. Matplotlib 사용하기

### 11.3.1. 점선 그래프 그리기

```python
data = np.random.randn(50).cumsum()
print(data)
```

```
[ 0.67951554  1.78671105 -0.8113654  -1.36233274 -0.60245961 -1.55807438
 -1.58010849  0.10710171 -0.65454452 -1.49792033 -4.0638153  -1.88842188
 -0.97047217 -0.00407407  0.74575524 -0.51028238  0.65518755  1.26795063
  0.42702737 -0.15854443 -1.20883959 -2.69517938 -1.8238372  -1.88806512
 -2.24162516 -1.13905281 -0.15094827 -0.80385505 -1.73113762 -2.08923885
 -1.17233306 -1.47571658  0.25418287  0.42281728 -0.35980094 -0.6916567
 -0.31161851  0.96175268  0.73279041 -0.76711256 -1.06476561 -1.29427003
 -0.87970066 -1.0092659  -1.91456048 -1.70267933 -2.25533234 -2.1789917
 -2.90443958 -2.50364733]
```



```python
plt.plot(data)
plt.show() # jupyter notebook에서는 show함수가 없어도 inline설정을 해주었기 때문에 그래프가 자동으로 출력된다.
```

![img](data:image/png;base64,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)



### 11.3.2. 여러 그래프 그릴 준비 하기

> subplot : 빈 그래프를 그리는 함수. plt.subplot(행, 열, 순서) 여기서 행과 열의 의미는 한 번에 보여줄 그래프의 행과 열을 의미하는 것이지 그래프 하나에 대한 행과 열 정보가 아니다.

```python
plt.subplot(3, 2, 1)
plt.subplot(3, 2, 2)
plt.subplot(3, 2, 3)
plt.subplot(3, 2, 4)
plt.subplot(3, 2, 5)
plt.subplot(3, 2, 6)
plt.show()
```

![img](data:image/png;base64,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)



### 11.3.3. Multi Graph 그리기

```python
hist_data = np.random.randn(100) # 히스토그램용 데이터
scat_data = np.arange(30) # 스캐터용 데이터

plt.subplot(2, 2, 1)
plt.plot(data)
plt.subplot(2, 2, 2)
plt.hist(hist_data, bins=20)
plt.subplot(2, 2, 3)
plt.scatter(scat_data, np.arange(30) + 3 * np.random.randn(30)) # plt.scatter(X축, Y축)
plt.show()
```

![img](data:image/png;base64,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)



### 11.3.4. 그래프 선 옵션

- 그래프를 그릴 때 표시되는 색이나 마커 패턴을 바꾸는 것
  - 색상 : b(파란색), g(초록색), r(빨간색), c(청록색), y(노란색), k(검은색), w(흰색)
  - 마커 : o(원), v(역삼각형), ^(삼각형), s(네모), +(플러스), .(점)
  - plt.plot(data, '`색상+마커`') 형태로 사용

```python
plt.plot(data, 'y--')
plt.show()
```

![img](data:image/png;base64,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)



## 11.4. 그래프 사이즈 조절

- plt.figure 안에 figsize를 이용하여 가로, 세로 길이 조절 가능 (inch 단위)

  ```python
  plt.figure(figsize=(10, 5)) # figsize=(가로, 세로)
  plt.plot(data, 'k+')
  plt.show()
  ```

  ![img](data:image/png;base64,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)

- 여러 그래프 그리고 그에 대한 크기 조절

  ```python
  plt.figure(figsize=(10, 5)) # 가장 위에 있어야 전체에 적용
  
  plt.subplot(2, 2, 1)
  plt.plot(data)
  plt.subplot(2, 2, 2)
  plt.hist(hist_data, bins=20)
  plt.subplot(2, 2, 3)
  plt.scatter(scat_data, np.arange(30) + 3 * np.random.randn(30))
  plt.show()
  ```

  ![img](data:image/png;base64,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)

- 비율 조절

  123



## 11.5. 그래프 겹치기 + legend 달기

```python
data = np.random.randn(30).cumsum()

plt.plot(data, 'k--')
plt.plot(data, 'k-', drawstyle='steps-post')
plt.show()
```

![img](data:image/png;base64,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)

subplot이 없기 때문에 한번에 그려진다.



```python
plt.plot(data, 'k--', label='Default')
plt.plot(data, 'k-', drawstyle='steps-post', label='steps-post')
plt.legend()
plt.show()
```

![img](data:image/png;base64,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)



## 11.6. 이름 달기

```python
plt.plot(np.random.randn(1000).cumsum())
plt.title('Random Graph')
plt.xlabel('Stages')
plt.ylabel('Values')
plt.show()
```

![img](data:image/png;base64,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)



## 11.7. 종합

```python
plt.title('Graph')
plt.plot(np.random.randn(1000).cumsum(), 'k^', label='one')
plt.plot(np.random.randn(1000).cumsum(), 'b.', label='two')
plt.plot(np.random.randn(1000).cumsum(), 'r', label='three')

plt.legend()
plt.show()
```

![img](data:image/png;base64,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)



## 11.8. 그래프 저장하기

```python
plt.savefig('saved_graph.svg') # 현재 작업환경에 저장
```

