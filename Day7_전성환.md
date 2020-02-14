# Day7_전성환

### 추억의 2048게임

```python
def move(dir):
    global N
    if dir == 'up':
        for j in range(N):
            s = 0
            cnt_i = 0
            for i in range(N):
                if arr[i][j] != 0:
                    if s != arr[i][j]:
                        pre = arr[i][j]
                        arr[i][j] = 0
                        arr[cnt_i][j] = pre
                        s = pre
                        cnt_i += 1
                    elif s == arr[i][j]:
                        arr[cnt_i - 1][j] = s * 2
                        arr[i][j] = 0
                        s = 0
    elif dir == 'down':
        for j in range(N):
            s = 0
            cnt_i = N-1
            for i in range(N-1, -1, -1):
                if arr[i][j] != 0:
                    if s != arr[i][j]:
                        pre = arr[i][j]
                        arr[i][j] = 0
                        arr[cnt_i][j] = pre
                        s = pre
                        cnt_i -= 1
                    elif s == arr[i][j]:
                        arr[cnt_i + 1][j] = s * 2
                        arr[i][j] = 0
                        s = 0

    elif dir == 'right':
        for i in range(N):
            s = 0
            cnt_j = N-1
            for j in range(N-1, -1, -1):
                if arr[i][j] != 0:
                    if s != arr[i][j]:
                        pre = arr[i][j]
                        arr[i][j] = 0
                        arr[i][cnt_j] = pre
                        s = pre
                        cnt_j -= 1
                    elif s == arr[i][j]:
                        arr[i][cnt_j + 1] = s * 2
                        arr[i][j] = 0
                        s = 0

    elif dir == 'left':
        for i in range(N):
            s = 0
            cnt_j = 0
            for j in range(N):
                if arr[i][j] != 0:
                    if s != arr[i][j]:
                        pre = arr[i][j]
                        arr[i][j] = 0
                        arr[i][cnt_j] = pre
                        s = pre
                        cnt_j += 1
                    elif s == arr[i][j]:
                        arr[i][cnt_j - 1] = s * 2
                        arr[i][j] = 0
                        s = 0


T = int(input())
dy = [-1, 0, 1, 0]
dx = [0, 1, 0, -1]
for case in range(1, T+1):
    N, command = input().split()
    N = int(N)
    arr = [list(map(int,input().split())) for _ in range(N)]
    move(command)
    print("#%d"%(case))
    for i in range(N):
        print(*arr[i])
```





### 농작물 수확하기

```python
T = int(input())
for case in range(1, T+1):
    N = int(input())
    arr = [list(map(int,list(input()))) for _ in range(N)]
    ans = 0
    for i in range(N//2+1):
        for j in range(N//2-i, N//2+1+i):
            ans += arr[i][j]
            
    for i in range(N//2+1, N):
        for j in range(-N//2+i+1, N+N//2-i):
            ans += arr[i][j]

    print("#%d"%(case), ans)
```





### 스도쿠 검증

```python
T = int(input())
for case in range(1, T+1):
    arr = [list(map(int,input().split())) for _ in range(9)]
    TF = False
    for i in range(9):
        lst = [False] * 9
        for j in range(9):
            val = arr[i][j]
            if lst[val-1] == True:
                TF = True
                break
            else:
                lst[val-1] = True
        if TF:
            break
    if TF:
        print("#%d" % (case), 0)
        continue

    for j in range(9):
        lst = [False] * 9
        for i in range(9):
            val = arr[i][j]
            if lst[val-1] == True:
                TF = True
                break
            else:
                lst[val-1] = True
        if TF:
            break
    if TF:
        print("#%d" % (case), 0)
        continue

    for i in range(3):
        for j in range(3):
            lst = [False] * 9
            for a in range(i*3, i*3+3):
                for b in range(j*3, j*3+3):
                    val = arr[a][b]
                    if lst[val - 1] == True:
                        TF = True
                        break
                    else:
                        lst[val - 1] = True
                if TF:
                    break
            if TF:
                break
        if TF:
            break

    if TF:
        print("#%d" % (case), 0)
    else:
        print("#%d"%(case), 1)
```





### 숫자 배열 회전

```python
T = int(input())
for case in range(1, T+1):
    N = int(input())
    arr = [list(map(int,input().split())) for _ in range(N)]
    print("#%d" % (case))
    for i in range(N):
        for j in range(N-1, -1, -1):
            print(arr[j][i], end='')
        print(end=' ')

        for j in range(N-1, -1, -1):
            print(arr[N-1-i][j], end='')
        print(end=' ')

        for j in range(N):
            print(arr[j][N-1-i], end='')
        print()
```





### Magnetic

```python
for case in range(1, 11):
    N = int(input())
    arr = [list(map(int,input().split())) for _ in range(N)]
    ans = 0
    for j in range(N):
        i = 0
        cnt = 0
        while i < N:
            if arr[i][j] == 1:
                cnt += 1
                s = 1
                break
            i += 1
            
        i += 1
        while i < N:
            if arr[i][j] != 0 and arr[i][j] != s:
                cnt += 1
                s = arr[i][j]
            i += 1

        ans += cnt//2

    print("#%d"%(case), ans)
```

