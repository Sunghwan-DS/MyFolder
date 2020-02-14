# Day 1

### 연습문제 2

```python
arr = [i+1 for i in range(10)]

n = len(arr)

for i in range(1<<n):
    lst = []
    for j in range(n+1):

        if i & (1<<j):
            lst.append(arr[j])

    if sum(lst) == 10:
        print(lst)
```





### 이진검색

```python
def binarySearch(a, low, high, key):
    if low > high:
        return False
    else:
        middle = (low + high) // 2
        if key == a[middle]:
            return True
        elif key < a[middle]:
            return binarySearch(a, low, middle-1, key)
        elif a[middle] < key:
            return binarySearch(a, middle+1, high, key)
```





### Selection Sort

```python
def selectionSort(a):
    for i in range(0, len(a)-1):
        min = i
        for j in range(i+1, len(a)):
            if a[min] > a[j]:
                min = j
        	a[i], a[min] = a[min], a[i]
```





### Sum

```python
for _ in range(10):
    T = int(input())
    field = []
    for i in range(100):
        field.append(list(map(int,input().split())))

    result = 0
    for i in range(100):
        x = (sum(field[i]))
        if x > result:
            result = x
    for j in range(100):
        y = 0
        for i in range(100):
            y += field[i][j]
        if y > result:
            result = y
    xy1 = 0
    for i in range(100):
        xy1 += field[i][i]
    if xy1 > result:
        result = xy1

    xy2 = 0
    for i in range(100):
        xy2 += field[i][99-i]
    if xy2 > result:
        result = xy2

    print("#%d %d"%(T, result))
```





### 연습문제 3

```python
def move(cnt, i, j, direction):
    global field
    if cnt == N**2 + 1:
        return
    field[i][j] = cnt
    if direction == 1:
        if j+1 <= N-1:
            if field[i][j+1] == 0:
                move(cnt+1, i, j+1, 1)
            else:
                move(cnt + 1, i+1, j, 2)
        else:
            move(cnt + 1, i + 1, j, 2)

    elif direction == 2:
        if i+1 <= N - 1:
            if field[i+1][j] == 0:
                move(cnt+1, i+1, j, 2)
            else:
                move(cnt + 1, i, j-1, 3)
        else:
            move(cnt + 1, i, j-1, 3)

    elif direction == 3:
        if j-1 >= 0:
            if field[i][j-1] == 0:
                move(cnt+1, i, j-1, 3)
            else:
                move(cnt + 1, i-1, j, 4)
        else:
            move(cnt + 1, i-1, j, 4)

    elif direction == 4:
        if field[i-1][j] == 0:
            move(cnt+1, i-1, j, 4)
        else:
            move(cnt+1, i, j+1, 1)


T = int(input())
for _ in range(T):
    N = int(input())
    field = [[0] * N for __ in range(N)]
    i = 0
    j = 0
    move(1, 0, 0, 1)
    print("#%d"%(_+1))
    for __ in range(N):
        print(" ".join([str(i) for i in field[__]]))
```



