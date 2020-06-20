# Section 1. 머신러닝의 개념과 용어

## 1. 기본적인 Machine Learning의 용어와 개념 설명

### 1.0. Basic concepts

- What is ML?
- What is learning?
  - supervised
  - unsupervised
- What is regression?
- What is classification?



### 1.1. Machine Learning

- Limitations of explicit programming (노골적인 프로그래밍의 한계들)
  - Spam filter: many rules
  - Automatic driving: too many rules
- Machine learning: "Field of study that gives computers the ability to learn without being explicitly programmed" Arthur Samuel (1959)
  - 일일이 프로그래밍 하지 말고 어떤 자료, 현상에서 자동적으로 배우면 어떨까? 
  - 프로그램 자체가 학습해서 무언가를 배우는 능력을 갖는 프로그램 = 머신러닝



### 1.2. Supervised/Unsupervised learning

- Supervised learning:
  - learning with labeled examples - training set
  - labeled 된 정해져있는 데이터를 이용해 학습시키는 방법
  - An example training set for four visual categories
  - cat, dog, mug, hat 이라고 label 된 데이터를 이용하여 학습시키는 것
- Unsupervised learning: un-labeled data
  - Google news grouping
  - Word clustering
    - 비슷한 단어들을 모으는 머신러닝 



### 1.3. Supervised learning

> 이번 강의에서 주로 다루게 될 학습 방법

- Most common problem type in ML
  - Image labeling: learning from tagged images
  - Email spam filter: learning from labeled (spam or ham) email
  - Predicting exam score: learning from previous exam score and time spent



### 1.4. Training data set

|      |  X   |      |  Y   |
| :--: | :--: | :--: | :--: |
|  3   |  6   |  9   |  3   |
|  2   |  5   |  8   |  2   |
|  2   |  3   |  5   |  1   |

Y: label

X: features

X와 Y를 이용하여 ML 학습

X_test = [9, 3, 6] 일 때 Y_test를 예상해주는 것



#### 1.4.1. AlphaGo

바둑 기보를 이용하여 A.G.를 학습시키고 이를 이용해 바둑을 둔다.

기존의 바둑 기보가 바로 training data set.



### 1.5. Types of supervised learning

- Predicting final exam score based on time spent
  - <strong>regression</strong>
- Pass/non-pass based on time spent
  - binary classification
- Letter grade (A, B, C, E and F) based on time spent
  - multi-label classification



#### 1.5.1. Predicting final exam score based on time spent

학습 시간과 점수의 관계

| x (hours) | y (score) |
| :-------: | :-------: |
|    10     |    90     |
|     9     |    80     |
|     3     |    50     |
|     2     |    30     |

Regression 모델이 있을 때, 학습 데이터(x, y)를 이용하여 학습시킨다. x = 7 일 때, y는 얼마의 값을 가질 것인가?



#### 1.5.2. Pass/non-pass based on time spent

| x (hours) | y (pass/fail) |
| :-------: | :-----------: |
|    10     |       P       |
|     9     |       P       |
|     3     |       F       |
|     2     |       F       |

오직 두 가지 평가 지표만을 가지고 평가. 즉, binary classification



#### 1.5.3. Letter grade (A, B, ...) based on time spent

| x (hours) | y (grade) |
| :-------: | :-------: |
|    10     |     A     |
|     9     |     B     |
|     3     |     D     |
|     2     |     F     |

여려 개의 평가 지표. multi classification



## 2. TensorFlow의 설치 및 기본적인 operations (new)

> TensorFlow: 구글에서 만든 딥러닝을 위한 오픈소스 라이브러리



### 2.1. TensorFlow

- TensorFlow<sup>TM</sup> is an open source software library for numerical computation using data flow graphs.
- Python!



### 2.2. What is a Data Flow Graph?

- Nodes in the graph represent mathematical operations
- Edges represent the multidimensional data arrays(tensors) communicated between them.



### 2.3. Installing TensorFlow

- Linux, Max OSX, Windows
  - (sudo -H) pip install --upgrade tensorflow
  - (sudo -H) pip install --upgrade tensorflow-gpu
- From source
  - bazel ...
- Google search/Community help
  - https://www.facebook.com/groups/TensorFlowKR/



#### 2.3.1. Check installation and version

```python
pip install --upgrade tensorflow-cpu==1.15.0 # v1.0
pip install --upgrade --pre tensorflow-cpu # v2.0
```

```python
import tencorflow as tf
tf.__version__
```

```
강의 기준: '1.0.0'
내 로컬: '2.2.0'
```





#### 2.3.2. 전체 소스코드

https://github.com/hunkim/DeepLearningZeroToAll/



### 2.4. TensorFlow Hello World! (v1.0)

```python
# Create a constant op
# This op is added as a node to the default graph
hello = tf.constant("Hello, TensorFlow!")

# seart a TF session
sess = tf.Session()

# run the op and get result
print(sess.run(hello))
```

```
b'Hello, TensorFlow!'
```

b'String' '<b>b</b>' indicates <i>Bytes literals</i>.

http://stackoverflow.com/questions/6269765/



#### 2.4.1. TensorFlow Hello World! (v2.0)

> 강의 내용에 해결 방안이 존재하지 않아 직접 조사한 내용

```python
import tensorflow as tf
msg = tf.constant('Hello, TensorFlow!')
tf.print(msg)
```

불편한 Session() 과정이 없어지고 바로 출력이 가능해졌다.



### 2.5. Computational Graph (v1.0)

```python
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly 암묵적으로 내포
node3 = tf.add(node1, node2) # node3 = node1 + node2 가능
```



```python
print("node1: ", node1, "node2: ", node2)
print("node3: ", node3)
```

```
node1:  tf.Tensor(3.0, shape=(), dtype=float32) node2:  tf.Tensor(4.0, shape=(), dtype=float32)
node3:  tf.Tensor(7.0, shape=(), dtype=float32)
```



```python
sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))
```

```
sess.run(node1, node2):  [3.0, 4.0]
sess.run(node3):  7.0
```





#### 2.5.1. Computational Graph (v2.0)

```python
tf.print("sess.run(node1, node2): ", [node1, node2])
tf.print("sess.run(node3): ", node3)
```

```
sess.run(node1, node2):  [3, 4]
sess.run(node3):  7
```



### 2.6. TensorFlow Mechanics

1. Build graph using TensorFLow operations
2. feed data and run graph (operation) sess.run (op)
3. update variables in the graph (and return values)



### 2.7. Placeholder (v1.0)

```python
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2, 4]}))
```

```
7.5
[ 3.  7.]
```



### 2.8. Everything is Tensor

```python
3 # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.] # a rank I tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
```

```
[[[1.0, 2.0, 3.0]], [[7.0, 8.0, 9.0]]]
```



### 2.9. Tensor Ranks, Shapes, and Types

```python
t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
```

| Rank | Math entity                      | Python example                                               |
| :--- | -------------------------------- | ------------------------------------------------------------ |
| 0    | Scalar (magnitude only)          | s = 483                                                      |
| 1    | Vector (magnitude and direction) | v = [1.1, 2.2, 3.3]                                          |
| 2    | Matrix (table of numbers)        | m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]                        |
| 3    | 3-Tensor (cube of numbers)       | t = [[[2], [4], [6]], [[8], [10], [12]], [[14], [16], [18]]] |
| n    | n-Tensor (you get the idea)      | ....                                                         |



| Rank | Shape               | Dimension number | Example                                 |
| ---- | ------------------- | ---------------- | --------------------------------------- |
| 0    | []                  | 0-D              | A 0-D tensor. A scalar                  |
| 1    | [D0]                | 1-D              | A 1-D tensor with shape[5]              |
| 2    | [D0, D1]            | 2-D              | A 2-D tensor with shape [3, 4]          |
| 3    | [D0, D1, D2]        | 3-D              | A 3-D tensor with shape [1, 4, 3]       |
| n    | [D0, D1, ..., Dn-1] | n-D              | A tensor with shape [D0, D1, ..., Dn-1] |



| Data type | Python type | Description            |
| --------- | ----------- | ---------------------- |
| DT_FLOAT  | tf.float32  | 32 bits floating point |
| DT_DOUBLE | tf.float64  | 64 bits floating point |
| DT_INT8   | tf.int8     | 8 bits signed integer  |
| DT_INT16  | tf.int16    | 16 bits signed integer |
| DT_INT32  | tf.int32    | 32 bits signed integer |
| DT_INT64  | tf.int64    | 64 bits signed integer |

