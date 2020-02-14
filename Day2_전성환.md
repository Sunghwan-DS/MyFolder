# Day2_전성환

### 금속막대

```python
def DFS(h, t, use_list):
    global result
    for j in range(N):
        if j not in use_list:
            if table[j*2] == t:
                pre = use_list[:]
                use_list.append(j)
                DFS(table[j*2], table[j*2+1], use_list)
                use_list = pre[:]
    if len(use_list) > len(result):
        result = use_list[:]


T = int(input())
for case in range(1, T+1):
    N = int(input())
    table = list(map(int,input().split()))
    result = []


    for i in range(N):
        use_list = [i]
        DFS(table[i*2], table[i*2+1], use_list)

    print("#%d"%(case)," ".join([str(table[i*2])+" "+str(table[i*2+1]) for i in result]))
```





### 색칠하기

```python
T = int(input())
for case in range(1, T+1):
    arr = [[1] * 10 for _ in range(10)]
    cnt = 0
    N = int(input())
    for _ in range(N):
        data = list(map(int,input().split()))
        for i in range(data[0], data[2]+1):
            for j in range(data[1], data[3]+1):
                arr[i][j] *= (data[4]+1)

    for i in range(10):
        for j in range(10):
            if arr[i][j] % 6 == 0:
                cnt += 1
    print("#%d"%(case), cnt)
```





### 부분집합의 합

```python
def check(lst):
    global result
    if len(lst) == N:
        if sum(lst) == K:
            result += 1
        return

    for i in range(lst[-1]+1, 13):
        pre = lst[:]
        lst.append(i)
        check(lst)
        lst = pre[:]

T = int(input())
for case in range(1, T+1):
    N, K = map(int,input().split())
    result = 0
    for i in range(1, 12):
        check([i])
    print("#%d"%(case), result)
```





### 이진탐색

```python
def check(high, low, num, idx):
    middle = (high + low) // 2
    if num > middle:
        return check(high, middle, num, idx+1)
    elif num < middle:
        return check(middle, low, num, idx+1)
    else:
        return idx

T = int(input())
for case in range(1, T+1):
    P, A, B = map(int,input().split())
    a = check(P, 1, A, 0)
    b = check(P, 1, B, 0)
    if a < b:
        print("#%d"%(case), "A")
    elif a > b:
        print("#%d"%(case), "B")
    else:
        print("#%d" % (case), "0")
```





### 특별한 정렬

```python
T = int(input())
for case in range(1, T+1):
    input()
    sequent = list(map(int,input().split()))
    sequent.sort()
    l = len(sequent)
    result = []
    TF = [-1, 0]
    k = 0
    while sequent:
        result.append(sequent.pop(TF[k%2]))
        k += 1

    print("#%d"%(case), *result[:10])
```

