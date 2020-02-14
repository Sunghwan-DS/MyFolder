# Day6_전성환

### 미로 1

```python
dy = [-1, 0, 1, 0]
dx = [0, 1, 0, -1]

for num in range(10):
    case = int(input())
    arr = [list(map(int,input())) for _ in range(16)]
    TF = False
    for i in range(16):
        for j in range(16):
            if arr[i][j] == 2:
                TF = True
                break
        if TF:
            break
    result = 0
    q = [[i, j]]
    visited = [[False] * 16 for _ in range(16)]
    visited[i][j] = True
    TF = False
    while q:
        current = q[-1]
        y = current[0]
        x = current[1]

        for dir in range(4):
            ny = y + dy[dir]
            nx = x + dx[dir]

            if 0 <= ny <= 15 and 0 <= nx <= 15:
                if arr[ny][nx] == 0 and not visited[ny][nx]:
                    q.append([ny, nx])
                    visited[ny][nx] = True
                    break

                elif arr[ny][nx] == 3:
                    q.append([ny, nx])
                    TF = True
                    result = 1
                    break

        else:
            q.pop(-1)

        if TF:
            break
    print("#%d %d"%(case, result))
```





### 종이붙이기

```python
T = int(input())

lst = [1,3]
for i in range(28):
    lst.append(lst[-1] + lst[-2] * 2)

for case in range(1, T+1):
    N = int(input())
    print("#%d"%(case), lst[N//10-1])
```





### 괄호검사

```python
T = int(input())
for case in range(1, T+1):
    sen = input()
    lst = []
    TF = True
    for l in sen:
        if l == '{':
            lst.append("{")
        elif l == '(':
            lst.append("(")
        elif l == '}':
            if lst == []:
                TF = False
                break
            if lst[-1] == "{":
                lst.pop(-1)
            else:
                TF = False
                break
        elif l == ')':
            if lst == []:
                TF = False
                break
            if lst[-1] == "(":
                lst.pop(-1)
            else:
                TF = False
                break

    if TF and lst == []:
        print("#%d"%(case), 1)
    else:
        print("#%d"%(case), 0)
```





### 그래프 경로

```python
T = int(input())
for case in range(1, T+1):
    V, E = map(int,input().split())

    node = [[] for _ in range(V+1)]

    for i in range(E):
        s, e = map(int,input().split())
        node[s].append(e)

    S, G = map(int,input().split())
    TF = False
    s = [S]
    visited = [S]
    while s:
        current = s[-1]

        for i in node[current]:
            if i == G:
                TF = True
                break

            elif i not in visited:
                s.append(i)
                visited.append(i)
                break

        else:
            s.pop()

        if TF:
            break

    if TF:
        print("#%d"%(case), 1)
    else:
        print("#%d" % (case), 0)
```





### 반복문자 지우기

```python
T = int(input())
for case in range(1, T+1):
    word = input()
    lst = []
    for i in word:
        if lst == []:
            lst.append(i)
        elif i == lst[-1]:
            lst.pop(-1)
        else:
            lst.append(i)

    print("#%d"%(case), len(lst))
```

