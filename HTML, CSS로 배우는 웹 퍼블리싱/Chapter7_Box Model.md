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





## 4. width, height

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
                /* 창의 가로 길이가 500px보다 작아지면 크기가 줄어들지 않는다. */
                min-width: 500px;
                /* 창의 세로 길이가 500px보다 커져도 크기가 늘어나지 않는다. */
                max-height: 500px;
            }
            
            .p2 {
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





## 5. width, height 정리

요소의 가로 길이(width)와 세로 길이(height)를 설정.

```CSS
p {
  border: 1px solid blue;
  width: 400px;
  height: 300px;
}
```



##### 이미지

사진의 크기도 똑같이 css에서 설정할 수 있다.

```CSS
.bond-img {
  width: 400px;
  height: 300px;
}
```



### 최소, 최대 가로 길이

`min-width`, `max-width`로 요소의 최소, 최대 가로 길이를 설정할 수 있다.

```CSS
.p1 {
  border: 1px solid blue;
  max-width: 1000px;
}

.p2 {
  border: 1px solid red;
  max-width: 200px;
}

.p3 {
  border: 1px solid blue;
  min-width: 2000px;
}

.p4 {
  border: 1px solid red;
  min-width: 200px;
}
```



### 최소, 최대 세로 길이

`min-height`, `max-height`로 요소의 최소, 최대 세로 길이를 설정할 수 있다.

```CSS
.p1 {
  border: 1px solid blue;
  min-height: 400px;
}

.p2 {
  border: 1px solid red;
  min-height: 200px;
}

.p3 {
  border: 1px solid blue;
  max-height: 1000px;
}

.p4 {
  border: 1px solid red;
  max-height: 50px;
}
```





## 6. overflow

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
                max-height: 200px;
                /* 넘친 부분 숨김 */
                overflow: hidden;
                /* 넘친 부분 보기: visible (기본값) */
                /* scroll : 스크롤로 넘친 부분 보기 */
                /* auto : 스크롤이 필요없으면 제거 */
            }
            
            .p2 {
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





## 7. overflow 정리

`width`, `height`, `max-width`, `max-height` 등을 설정하다 보면 내용물이 들어갈 공간이 부족한 경우가 있다.

```css
p {
  border: 1px solid blue;
  width: 300px;
  height: 200px;
}
```

이렇게 넘쳐나는 내용물을 `overflow` 속성으로 처리해줄 수 있는 몇 가지 옵션이 있다.



### 옵션 1: visible

`visible` 값을 사용하면 넘쳐나는 내용물이 그대로 보이며, 따로 설정해주지 않으면 이게 기본값이다.

```css
p {
  border: 1px solid blue;
  width: 300px;
  height: 200px;
  overflow: visible;
}
```



### 옵션 2: hidden

`hidden` 값을 사용하면 넘쳐나는 부분을 아예 숨겨줄 수도 있다.

```css
p {
  border: 1px solid blue;
  width: 300px;
  height: 200px;
  overflow: hidden;
}
```



### 옵션 3: scroll

내용물을 숨겼다가, 사용자가 스크롤을 하면 볼 수 있게 해주는 방법도 있다.

```css
p {
  border: 1px solid blue;
  width: 300px;
  height: 200px;
  overflow: scroll;
}
```



### 옵션 4: auto

`scroll`과 거의 똑같은데, 한 가지 차이점이 있다. `scroll`은 **항상** 스크롤바를 보여주고, `auto`는 **내용물이 넘쳐날 때만** 스크롤바를 보여준다.

참고로 Mac OS에서는 스크롤을 할 때만 스크롤바를 보여주는 경향이 있기 때문에 `scroll`과 `auto`의 차이를 보기 힘들 수도 있다.

```css
p {
  border: 1px solid blue;
  width: 300px;
  height: 200px;
  overflow: auto;
}
```





## 8. border

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
                border: 2px solid red;
                /* dotted: 점선, dashed: 파선 */
            }
            
            .p2 {
                border-top: 3px dotted #4d9fff;
                border-bottom: 2px dashed red;
                border-left: 5px solid green;
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





## 9. border 정리

다른 속성들과 마찬가지로, 테두리를 설정해주는 방법도 다양하다.



### 한 줄에 끝내기

가장 일반적인 방법은 `border` 속성으로 한 줄에 다 쓰는 것이다. 이 방식을 사용하면 위, 아래, 왼쪽, 오른쪽 모두 같은 테두리가 생긴다. 값을 쓰는 순서는 굵기, 스타일(실선, 점선 등), 색이다.

```css
.p1 {
  border: 2px solid #4d9fff;
}

.p2 {
  border: 2px dotted #4d9fff;
}

.p3 {
  border: 2px dashed #4d9fff;
}
```



### 명확하게 나누기

다른 방법은 `border-style`, `border-color`, `border-width` 속성을 써서 테두리의 스타일을 하나씩 지정해주는 것이다.

```css
.p1 {
  border-style: dotted;
  border-color: red;
  border-width: 5px;
}
```



### 다채로운 테두리

지금까지는 4면의 테두리가 모두 같았는데, 다 다르게 설정해주고 싶으면 이렇게 하면 된다.

```css
.p1 {
  border-top: 3px dotted #4d9fff;
  border-bottom: 2px dashed red;
  border-left: 5px solid green;
}
```



### 테두리 없애기

어떤 요소들(예: `<input>` 태그)은 기본적으로 테두리가 설정되어 있다. 이런 요소들의 테두리를 없애고 싶으면 직접 `border` 속성을 설정해주면 되는데, 두 가지 방법이 있다.

1. border: none;
2. border: 0;