# Day3_전성환

### 정곤이의 단조 증가하는 수

```python
T = int(input())
for case in range(1, T+1):
    N = int(input())
    num = list(map(int,input().split()))
    result = -1
    for i in range(N-1):
        for j in range(i+1, N):
            value = num[i] * num[j]
            if value <= result:
                continue
            word = str(value)
            if len(word) == 1:
                if value > result:
                    result = value
            else:
                TF = False
                for _ in range(len(word)-1):
                    if word[_] > word[_+1]:
                        break
                else:
                    TF = True
                if TF:
                    if value > result:
                        result = value
    print("#%d %d"%(case, result))
```





### 다솔이의 다이아몬드 장식

```pyhton
T = int(input())
for case in range(1, T+1):
    word = input()
    l = len(word)
    print("..#.."+".#.."*(l-1))
    print(".#.#."+"#.#."*(l-1))
    print("#.%s.#"%(word[0]),end="")
    for i in range(1, l):
        print(".%s.#"%(word[i]),end="")
    print("\n.#.#."+"#.#."*(l-1))
    print("..#.."+".#.."*(l-1))
```





### 영준이의 카드 카운팅

```python
T = int(input())
for case in range(1, T+1):
    card = input()
    card_lst = [0, 0, 0, 0]
    check = []
    for i in range(len(card) // 3):
        j = card[3*i:3*i+3]
        if j in check:
            print("#%d ERROR" % (case))
            break
        else:
            check.append(j)
            if j[0] == "S":
                card_lst[0] += 1
            elif j[0] == "D":
                card_lst[1] += 1
            elif j[0] == "H":
                card_lst[2] += 1
            elif j[0] == "C":
                card_lst[3] += 1

    else:
        print("#%d %d %d %d %d"%(case, 13 - card_lst[0], 13 - card_lst[1], 13 - card_lst[2], 13 - card_lst[3]))
```





### 파스칼의 삼각형

```python
T = int(input())
for case in range(1, T+1):
    N = int(input())
    print("#%d" % (case))
    print("1")
    lst = [1]
    for i in range(N-1):
        pre_lst = lst[:]
        lst = [1]
        for j in range(len(pre_lst)-1):
            lst.append(pre_lst[j]+pre_lst[j+1])
        lst.append(1)
        print(*lst)
```





### 스위치 켜고 끄기

```python
def change(state):
    if state == 0:
        return 1
    elif state == 1:
        return 0

N = int(input())
bulb = list(map(int, input().split()))
student = int(input())
for _ in range(student):
    s, card = map(int, input().split())
    if s == 1:
        for index in range(N):
            if (index+1) % card == 0:
                bulb[index] = change(bulb[index])

    elif s == 2:
        max_range = min(card-1, N-card)
        bulb[card-1] = change(bulb[card-1])
        for check in range(1, max_range+1):
            if bulb[card - 1 -check] == bulb[card - 1 + check]:
                bulb[card - 1 - check] = change(bulb[card - 1 - check])
                bulb[card - 1 + check] = change(bulb[card - 1 + check])
            else:
                break

T = N // 20
for i in range(T):
    print(" ".join([str(i) for i in bulb[i*20:(i+1)*20]]))
print(" ".join([str(i) for i in bulb[T*20:]]))
```

