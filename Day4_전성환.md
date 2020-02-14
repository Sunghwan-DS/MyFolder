# Day4_전성환

### 작업순서 입력

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





### ladder1

```python
for case in range(1, 11):
    input()
    arr = [list(map(int,input().split())) for _ in range(100)]
    for x in range(100):
        if arr[99][x] == 2:
            break

    y = 99
    while y != 0:
        y -= 1
        TF=True
        if 0 <= x-1:
            if arr[y][x-1] == 1:
                TF=False
                while True:
                    x -= 1
                    if not (0 <= x-1):
                        break
                    if arr[y][x-1] != 1:
                        break

        if TF and x+1 <= 99:
            if arr[y][x+1] == 1:
                while True:
                    x += 1
                    if not (x+1 <= 99):
                        break
                    if arr[y][x+1] != 1:
                        break

    print("#%d %d"%(case, x))
```





### 파리퇴치

```python
T = int(input())
for case in range(1, T+1):
    N, M = map(int,input().split())
    arr = [list(map(int,input().split())) for _ in range(N)]
    result = 0
    for i in range(N-M+1):
        for j in range(N-M+1):
            value = 0
            for a in range(i, i+M):
                for b in range(j, j+M):
                    value += arr[a][b]
            if result < value:
                result = value

    print("#%d %d"%(case, result))
```





### 의석이의 세로로 말해요

```python
T = int(input())
for case in range(1, T+1):
    arr = [input()+" "*15 for _ in range(5)]
    result = ""
    j = 0
    while True:
        str = ""
        for i in range(5):
            if arr[i][j] != " ":
                str += arr[i][j]
        if str != "":
            result += str
            j += 1
        else:
            break
    print("#%d"%(case), result)
```





### 오셀로

```python
def go(x, y, c):
    lst = [[] for _ in range(8)]
    check = [False] * 8
    arr[y][x] = c

    for d in range(1, N):
        for dir in range(8):
            if not check[dir]:
                ny = y + d * dy[dir]
                nx = x + d * dx[dir]
                if 0 <= nx <= N-1 and 0 <= ny <= N-1:
                    if arr[ny][nx] == c:
                        check[dir] = True
                    elif arr[ny][nx] == 0:
                        lst[dir] = []
                        check[dir] = True
                    else:
                        lst[dir].append([ny, nx])

    for dir in range(8):
        if check[dir]:
            for co in lst[dir]:
                arr[co[0]][co[1]] = c


T = int(input())

dy = [0, 0, 1, 1, 1, -1, -1, -1]
dx = [-1, 1, -1, 0, 1, -1, 0, 1]

for case in range(1, T+1):
    N, M = map(int,input().split())
    arr = [[0]*N for _ in range(N)]
    arr[N//2-1][N//2-1] = 2         # 흑 = 1, 벡 = 2
    arr[N // 2][N // 2] = 2
    arr[N // 2 - 1][N // 2] = 1
    arr[N // 2][N // 2 - 1] = 1

    for i in range(M):
        x, y, c = map(int,input().split())
        go(x-1, y-1, c)

    cnt1 = 0
    cnt2 = 0

    for i in range(N):
        for j in range(N):
            if arr[i][j] == 1:
                cnt1 += 1
            elif arr[i][j] == 2:
                cnt2 += 1

    print("#%d %d %d"%(case, cnt1, cnt2))
```

