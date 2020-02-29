# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 7. Box Model

## 1. Box Model 소개

문단을 오른쪽 클릭하고 검사를 누르면 사각형 즉, 박스형태를 확인할 수 있다.

![image-20200301000850690](C:\Users\전재인\AppData\Roaming\Typora\typora-user-images\image-20200301000850690.png)



```html
<!DOCTYPE html>

<html>
    <head>
        <title>Box Model</title>
        <meta charset="utf-8">
        <style>
            h1 {
                /* 테투리에 5px 두께의 빨간 박스 */
                border: 5px solid red;
                width: 500px;
                height: 300px;
            }
            
            .p1 {
                border: 5px solid red;
             	margin-top: 100px;
                margin-bottom: 100px;
                margin-left: 50px;
            }
            
            .p2 {
                border: 5px solid red;
                padding-top: 50px;
                padding-bottom: 50px;
                padding-left: 50px;
                padding-right: 50px;
            }
        </style>
    </head>
    
    <body>
        <h1>Life is Beautiful</h1>
        
        <p class="p1">
            문단1 문단1 문단1 문단1 문단1 문단1 문단1 문단1 문단1
        </p>
        
        <p class="p2">
            문단2 문단2 문단2 문단2 문단2 문단2 문단2 문단2 문단2
        </p>
    </body>
</html>
```





## 2. margin, padding

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Box Model</title>
        <meta charset="utf-8">
        <style>
            h1 {
                border: 5px solid red;
            }
            
            .p1 {
                border: 5px solid red;
                /*
             	padding-top: 50px;
                padding-bottom: 20px;
                padding-left: 80px;
                padding-right: 65px;
                */
                padding: 50px 65px 20px 80px; /* 위부터 시계방향 */
            }
            
            .p2 {
                /*
                border: 5px solid red;
                margin-top: 50px;
                margin-bottom: 50px;
                margin-left: 50px;
                margin-rignt: 50px;
                */
                margin: 50px;
            }
        </style>
    </head>
    
    <body>
        <h1>Life is Beautiful</h1>
        
        <p class="p1">
            문단1 문단1 문단1 문단1 문단1 문단1 문단1 문단1 문단1
        </p>
        
        <p class="p2">
            문단2 문단2 문단2 문단2 문단2 문단2 문단2 문단2 문단2
        </p>
    </body>
</html>
```





## 3. margin & padding 정리

요소는 내용(content), 패딩(padding), 테두리(border)로 이루어져 있다. Padding은 내용과 테두리 사이의 '여유 공간'이지만 반면에 Margin은 요소 주위의 여백이다. 즉, 테두리 밖의 공간.



### Padding

Padding을 주는 몇 가지 방법



#### 가장 직관적인 방법

```CSS
p {
  border: 1px solid blue;
  padding-top: 20px;
  padding-bottom: 40px;
  padding-left: 120px;
  padding-right: 60px;
}
```



#### 한 줄로 쓰는 법

한 줄에 쓰고 싶으면 `padding` 속성을 쓰면 되며. 순서는 위(`padding-top`)부터 시계 방향으로 하나씩 쓰면 된다.

```CSS
/*
p {
  padding: 위 오른쪽 아래 왼쪽;
}
*/
p {
  border: 1px solid blue;
  padding: 20px 60px 40px 120px;
}
```



#### 위, 아래, 왼쪽, 오른쪽이 다 같은 경우

만약 위, 아래, 왼쪽, 오른쪽에 똑같은 padding을 주고 싶으면 더 간편하다. 모두 `50px`의 padding을 주려면 이렇게 하면 된다.

```CSS
p {
  border: 1px solid blue;
  padding: 50px;
}
```



#### 위, 아래가 같고, 왼쪽, 오른쪽이 같은 경우

위, 아래에 `50px`의 padding을 주고, 왼쪽, 오른쪽에 `100px`의 padding을 주려면 이렇게 하면 된다.

```CSS
p {
  border: 1px solid blue;
  padding: 50px 100px;
}
```



### Margin

요소에게 margin을 주는 방법은 padding을 주는 방법과 똑같습니다.



#### 가운데 정렬

요소를 가운데 정렬하고 싶으면 왼쪽과 오른쪽 `margin` 값을 `auto`로 설정해줘야 한다. `auto`는 말 그대로 '자동으로 계산'하라는 뜻으로 왼쪽과 오른쪽을 `auto`로 설정하면 자동으로 왼쪽과 오른쪽을 똑같이 함으로써 요소는 가운데 정렬이 된다.

```CSS
p {
  border: 1px solid blue;
  width: 500px;
  margin-left: auto;
  margin-right: auto;
}
```

