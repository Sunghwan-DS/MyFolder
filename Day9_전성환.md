# Day9_전성환

### 스택1 연습3

```python
road = [[], [2,3], [1,4,5], [1,7], [2,6], [2,6], [4,5,7], [3,6]]

stack = [1]
visited = [0] * 8
visited[1] = 1

while stack:
    print(stack)
    current = stack[-1]
    for next in road[current - 1]:
        if next not in visited:
            stack.append(next)
            visited.append(next)
            break

    else:
        stack.pop()
```





### 미로

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





### atoi

```python
def atoi(strr):
    n=len(strr)
    ans=0
    for idx,i in enumerate(strr):
        ans += (ord(i)-48)*(10**(n-1-idx))
    return ans
```





### itoa

```python
def itoa(intt):
    a=''
    a=str(intt%10)+a
    intt=intt//10
    while intt!=0:
        a=str(intt%10)+a
        intt=intt//10
    return a
```





### GNS

```python
T = int(input())

for case in range(1, T+1):
    INFO = [[[0] * 128 for _ in range(128)] for __ in range(128)]
    input()
    words = list(input().split())
    for word in words:
        INFO[ord(word[0])][ord(word[1])][ord(word[2])] += 1

    print("#%d"%(case))
    ans = ""
    ans += "ZRO " * INFO[ord('Z')][ord('R')][ord('O')]
    ans += "ONE " * INFO[ord('O')][ord('N')][ord('E')]
    ans += "TWO " * INFO[ord('T')][ord('W')][ord('O')]
    ans += "THR " * INFO[ord('T')][ord('H')][ord('R')]
    ans += "FOR " * INFO[ord('F')][ord('O')][ord('R')]
    ans += "FIV " * INFO[ord('F')][ord('I')][ord('V')]
    ans += "SIX " * INFO[ord('S')][ord('I')][ord('X')]
    ans += "SVN " * INFO[ord('S')][ord('V')][ord('N')]
    ans += "EGT " * INFO[ord('E')][ord('G')][ord('T')]
    ans += "NIN " * INFO[ord('N')][ord('I')][ord('N')]

    print(ans[:-1])
```





### 문자열비교

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

