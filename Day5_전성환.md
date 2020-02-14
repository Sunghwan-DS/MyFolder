# Day5_전성환

### 선택정렬_recursion

```python
def GetSome(here, end):
    # 3. basecase 작성, 2번에서 작성한 바뀐 크기를 근거로
    if here >= end: return
    # 1. 무엇이 반복되고 있는가
    minmin = Data[here]
    where = here
    for now in range(here, end+1):
        if Data[now] < minmin:
            minmin = Data[now]
            where = now
        Data[here], Data[where] = Data[where], Data[here]
    # 2. 크기를 바꾸어서 재귀로 호출
    GetSome(here+1, end)

GetSome(0, 4)
print(Data)
```





### ladder1_recursion

```python
def go(x, y):
    global case
    if y == 0:
        return print("#%d %d"%(case, x))

    TF = True
    if 0 <= x - 1:
        if arr[y][x - 1] == 1:
            TF = False
            while True:
                x -= 1
                if not (0 <= x - 1):
                    break
                if arr[y][x - 1] != 1:
                    break

    if TF and x + 1 <= 99:
        if arr[y][x + 1] == 1:
            while True:
                x += 1
                if not (x + 1 <= 99):
                    break
                if arr[y][x + 1] != 1:
                    break
    go(x,y-1)

for case in range(1, 11):
    input()
    arr = [list(map(int,input().split())) for _ in range(100)]
    for x in range(100):
        if arr[99][x] == 2:
            break
    go(x, 99)
```





### 작업순서 입력_recursion

```python
def DFS(num, arr, use_lst):
    global V, TF, case
    if TF:
        return
    if arr[num-1] == []:
        new_arr = [arr[i][:] for i in range(V)]
        new_use = use_lst[:]
        new_use.append(num)

        if len(new_use) == V:
            print("#%d"%(case),*new_use)
            TF = True
            return

        for i in range(V):
            if num in new_arr[i]:
                new_arr[i].remove(num)

        for i in range(V):
            if i+1 in new_use:
                pass
            else:
                DFS(i+1, new_arr, new_use)
    else:
        return


for case in range(1, 11):
    V, E = map(int,input().split())
    arr_init = [[] for _ in range(V)]

    table = list(map(int,input().split()))
    for n in range(E):
        arr_init[table[n*2+1]-1].append(table[n*2])

    start_lst = []
    for i in range(V):
        if sum(arr_init[i]) == 0:
            start_lst.append(i+1)

    TF = False
    for _ in start_lst:
        DFS(_, arr_init, [])
```



