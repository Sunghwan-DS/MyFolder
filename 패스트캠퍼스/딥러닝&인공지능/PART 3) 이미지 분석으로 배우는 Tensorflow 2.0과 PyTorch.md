# 1. 개요

## 1.1. 목적

- 간단한 기본 예시 데이터로 학습 말고, 실전에서 사용할 수 있는 TensorFlow와 PyTorch 사용
- 모델 성능 향상에 필요한 추가 기본 방법
- 캐글이나 실전에서 필요한 방법들 소개



## 1.2. 강의 구성

- 예제가 아닌 custom한 데이터를 넣는 방법
- 이미지 전처리 (Augmentation) 하는 방법
- Callbacks를 통해 TensorBoard, Learning Rate Schedule, Checkpoint
- 모델 저장 및 로드



### 1.2.1. Custom한 데이터 넣는 방법

엄청 고급적인 기술이거나 어려운 기술은 아니지만, 앞에서는 예제로라도 모델을 돌릴 수 있는 수준

이번에는 Custom한 데이터이든 캐글에서든 데이터를 받게 되면 직접 모델에 적용해볼 수 있는 비교적 실전형



### 1.2.2. 이미지 전처리

요리를 할 때 재료를 다듬어서 후라이팬이나 조리기에 넣는 듯 데이터도 다듬어서 넣어야 함



### 1.2.3. Augmentation

데이터를 증폭시켜 다양하게 이미지에 변화주어 모델에 적용

Augmentation을 통해 여러 환경에서도 적응이 되도록 모델에게 하드 트레이닝



### 1.2.4. Callbacks

모델이 학습 도중 Epoch 또는 Step 단위로 이벤트를 일으키는 옵션

- TensorBoard는 완전 실시간이 아니라 정해진 타이밍에 실행시켜 기록을 담음
- Learning Rate Schedule. Learning Rate을 하나로만 고정하는 것이 아닌 진행할 때 마다 LR을 줄여 Loss가 튀는 것을 방지
- Checkpoint. 모델 학습 도중 학습된 Weight를 저장. 기본으로는 매 Epoch 마다 저장하고 이때 성능이 향상 되었을 때만 저장 가능



### 1.2.5. 모델 저장 및 불러오기

앞서 배운 Weight만 저장하는 것과는 다르게 모델 구조 및 optimizer도 저장 가능

학습 도중에는 Model이나 optimizer까지 저장할 필요는 없음

Model과 otimizer도 함께 저장해두면 나중에 이어서 학습하기 용이

Weight만 저장하여 Transfer Learning에도 적용 가능



# 2. 데이터 준비하기

```python
import os
from glob import glob

import numpy as np

import tensorflow as tf
from PIL import Image

import matplotLib.pyplot as plt
%matplotlib inline
```

```
os.getcwd() # 현재 경로
```

```python
os.listdir('dataset/mnist_png/')
```

```python
data_paths = glob('dataset/mnist_png/training/0/*.png')
```

```python
data_paths[0]
len(data_paths)
```



## 2.1. 데이터 분석 (MNIST)