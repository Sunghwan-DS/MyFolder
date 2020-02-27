# Day10_전성환

### 수의 새로운 연산

```python
T = int(input())
for case in range(1, T+1):
    p, q = map(int,input().split())
    n=1
    while p > n * (n+1) // 2:
        n += 1
    n -= 1
    x_p = p - n*(n+1)//2
    y_p = n + 2 - x_p

    n = 1
    while q > n * (n + 1) // 2:
        n += 1
    n -= 1
    x_q = q - n * (n + 1) // 2
    y_q = n + 2 - x_q

    n = x_p+y_p+x_q+y_q-2
    print("#%d"%(case),n*(n+1)//2 + x_p + x_q)
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





### 추억의 2048

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





### 준혁이 여자친구

```python
N, M = map(int,input().split())
price = [[0] * (N+1) for _ in range(N+1)]
min_val = [999999] * (N + 1)
for i in range(M):
    a, b, val = map(int,input().split())
    price[a][b] = val
    queue = [(1, 0)]
    min_val[1] = 0
while queue:
    current, p = queue.pop(0)
    if p > min_val[current]:
        continue
    for arrive, val in enumerate(price[current]):
        new_val = p + price[current][arrive]
        if val and min_val[arrive] > new_val:
            min_val[arrive] = new_val
            queue.append((arrive, new_val))
print(min_val[7])
```





### 문자열 비교 (Bruteforce)

```python
T = int(input())
for case in range(1, T+1):
    s1 = input()
    s2 = input()
    if s1 in s2:
        print("#%d"%(case), 1)
    else:
        print("#%d"%(case), 0)
```





### 문자열 비교 (KMP)

```python
def make_pi(N):
    m = len(N)
    pi = [0] * m        # 어느 지점에서 만들어지는 중복된 최장 문자열
    begin = 1
    matched = 0
    while begin + matched < m:      # begin + match => 검토하고자 하는 현재 위치
        if N[begin + matched] == N[matched]:
            matched += 1
            pi[begin + matched - 1] = matched       # 검토한 자리(begin + match)에 저장하고 싶은데 match가 이미 1 올랐으므로 begin + mathch - 1
        else:
            if matched == 0:        # match 된 적이 없으면 begin만 + 1
                begin += 1
            else:
                begin += matched - pi[matched - 1]      # 검토할 자리를 얼마나 이동할 것인가? => 현재 매칭값 - 이전의 최장 길이
                matched = pi[matched - 1]       # 현재 매칭값은 이전의 최장 중복 문자열로 이동
    return pi


def kmp(sen, pat):
    n = len(sen)
    m = len(pat)
    pi = make_pi(pat)
    begin = 0
    matched = 0
    ans = 0
    while begin <= n - m:
        if matched < m and sen[begin + matched] == pat[matched]:
            matched += 1
            if matched == m:
                ans = 1
                break
        else:
            if matched == 0:
                begin += 1
            else:
                begin += matched - pi[matched - 1]
                matched = pi[matched - 1]
    return ans

T = int(input())
for case in range(1, T+1):
    pattern = input()
    sentence = input()
    print("#%d" % (case), kmp(sentence, pattern))
```

