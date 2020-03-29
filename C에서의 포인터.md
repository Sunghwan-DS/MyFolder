# C에서의 포인터

## 1. 연산자의 이해

```c
#include <stdio.h>

int main(void) {
	int* pnum; // 포인터 변수 pnum 선언
	int num = 12345; // num값 초기화
	pnum = &num; // pnum에 num의 주소값을 저장
	printf("num의 값 : %d\n", num);
	printf("pnum이 가리키는 변수의 값 : %d\n", *pnum); // num을 출력한 것과 결과가 같다.
	printf("pnum이 저장한 주소 값 : %d\n", pnum);
	
    return 0;
}
```



`*` : 주소 값을 주소에 저장된 값으로 바꾸는 연산. 포인터가 가리키는 메모리 공간에 접근하는 것.

`&` : 값이 저장된 변수명으로부터 주소를 불러오는 연산.  (`*`와 반대의 개념이라 생각하면 이해하기 쉽다.)



### 1.1 int, char, double 등은 무엇일까?

파이썬에서는 변수형을 먼저 지정하지 않는다. C에서 이와 같은 행위는 무엇이라 생각하면 좋을까? 내가 이것을 변수에 주소를 할당하는 행위라고 이해하였다.

```c
int num; // num이란 변수는 선언되기 전 값을 어디에 저장할지 정해두지 않았다.
num = 10;
```

num이란 변수는 선언되기 전 값을 어디에 저장할지 정해두지 않았다. 때문에 int num; 이라고 선언되는 순간 어떠한 주소지(예컨데 0x00010)에 연결이 되는 것이다. num이 선언된 후에는 num은 값을 저장하고 있지는 않지만 주소지는 보유한 상태이기 때문에 `pnum = &num;` 과 같이 주소지를 포인트변수 pnum에 저장할 수 있는 것이다.



## 2. 포인터와 배열

### 2.1 arr이 갖는 의미 (vs 파이썬)

흔히 배열의 이름으로 사용되는 arr은 `arr[10]` 혹은 `arr[10][10]` 처럼 사용된다. 파이썬에서는 arr을 그냥 출력한다던지 혹은 sum(arr) 등이 가능하였다. arr이 배열 전체를 가르키는 것처럼 사용된 것이다. 그러나 C에서 arr은 주소 값을 의미하며 그 위치는 arr[0] 값이 저장된 곳이다. 즉 arr == &arr[0] 이라고 할 수 있다. 또한 arr부터 배열의 길이 만큼 arr+i == &arr[i] 를 만족한다.



### 2.2 포인터 변수로 이뤄진 배열: 포인터 배열

##### 포인터 배열 선언방식

```c
int * arr1[20]; // 길이가 20인 int형 포인터 배열 arr1
double * arr2[30]; // 길이가 30인 double형 포인터 배열 arr2
```



```c
#include <stdio.h>

int main(void){
    int num1 = 10, num2 = 20, num3 = 30;
    int *arr[3] = {&num1, &num2, &num3};
    
    // arr[0], arr[1], arr[2]에 저장된 값도 주소이므로 이 주소를 이용해 값을 불러오기 위해 * 을 붙힌다.
    printf("%d \n", *arr[0]);
    printf("%d \n", *arr[1]);
    printf("%d \n", *arr[2]);
    
    return 0;
}
```

##### 실행결과

```
10
20
30
```





## 3. 포인터와 함수에 대한 이해

### 3.1 함수의 인자로 배열 전달하기

#### 3.1.1 인자전달의 기본방식은 값의 복사이다!

```c
int SimpleFunc(int num) {}
int main(void){
    int age=17;
    SimpleFunc(age); // age에 저장된 값이 매개변수 num에 복사됨
    ...
}
```

위 코드의 SimpleFunc 함수의 호출을 통해서 인자로 age를 전달하고 있다. 그러나 <strong>실제로 전달되는 것은 age가 아닌, age에 저장된 값이다!</strong> 그리고 그 값이 매개변수 num에 복사되는 것이다.

내가 넘겨준 age라는 값을 함수에서 지역 변수 num을 새로 선언하고 새로 선언된 num의 주소지를 넘겨받은 age라는 값으로부터 주소지를 얻어 num 주소지에 연결한다고 생각하면 된다.



```c
#include <stdio.h>

void ShowArayElem(int * param, int len){
    int i;
    for (i=0; i<len; i++)
        printf("%d ", param[i]);
    printf("\n");
}
    
int main(void){
    int arr1[3]={1, 2, 3};
    int arr2[5]={4, 5, 6, 7, 8};
    ShowArayElem(arr1, sizeof(arr1) / sizeof(int));
    ShowArayElem(arr2, sizeof(arr2) / sizeof(int));
    return 0;
}
```

##### 실행결과

```
1 2 3
4 5 6 7 8
```



### 3.2 포인터 대상의 const 선언

#### 3.2.1 포인터 변수가 참조하는 대상의 변경을 허용하지 않는 const 선언

```c
int main(void){
    int num=20;
    const int * ptr=&num;
    *ptr=30; // 컴파일 에러!
    num=40; // 컴파일 성공!
}
```

<strong>포인터 변수 ptr을 이용해서 ptr이 가리키는 변수에 저장된 값을 변경하는 것을 허용하지 않는다.</strong>

그렇다고 해서 포인터 변수 ptr이 가리키는 변수 num이 상수화되는 것은 아니다. 따라서 다음과 같이 변수 num에 저장된 값을 변경하는 것은 허용이 된다.