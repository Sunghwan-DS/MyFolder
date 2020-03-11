# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 7. Box Model

> 예쁜 디자인 뒤에는 보이지 않는 많은 숫자들이 있습니다. 가로, 세로, 여백, 간격 등이 어떻게 적용되는지 배워보고, 웹페이지의 레이아웃을 자유자재로 조정해보세요.

## 1. Box Model 소개

문단을 오른쪽 클릭하고 검사를 누르면 사각형 즉, 박스형태를 확인할 수 있다.

![image-20200311111004653](upload\image-20200311111004653.png)



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





## 12. 박스 꾸미는 몇 가지 방법

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Box Model</title>
        <meta charset="utf-8">
        <style>
            body {
                background-color: gray;
            }
            
            .p1 {
                width: 400px;
                border: 5px solid green;
                padding: 50px;
                border-radius: 50px;
                background-color: transparent;
            }
            
            .p2 {
                width: 400px;
                padding: 50px;
                box-shadow: 10px 10px 50px 10px red;
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





## 13. 둥근 모서리

`border-radius`라는 속성을 사용하면 요소의 모서리를 둥글게 만들 수 있는데, 더 큰 값을 쓸수록 더 둥글게 된다.

```CSS
.div1 {
  border: 1px solid green;
  border-radius: 5px;
  margin-bottom: 20px;
}

.div2 {
  border: 1px solid green;
  border-radius: 30px;
}
```



### 개별 설정

그냥 `border-radius` 속성을 사용하면 모서리 네 개 모두 똑같이 둥글게 되는데, 각 모서리를 개별 설정할 수도 있다.

```CSS
h1 {
  border: 1px solid green;
  border-top-left-radius: 50px; /* 왼쪽 위 */
  border-top-right-radius: 5px; /* 오른쪽 위 */
  border-bottom-right-radius: 0px; /* 오른쪽 아래 */
  border-bottom-left-radius: 20px; /* 왼쪽 아래 */
}
```





## 14. 배경색

### 배경색

`background-color` 속성을 사용하면 배경색을 설정할 수 있다. 폰트 색을 설정할 때처럼 색 이름, RGB 코드, HEX 코드 중 하나를 입력하면 된다.

```CSS
h1 {
  background-color: #4d9fff;
}
```



#### 페이지 배경색

페이지 전체의 배경색을 설정하고 싶으면 `body` 태그에 `background-color` 속성을 입혀주면 된다.

그리고 배경색을 투명하게 두고 싶으면 `transparent` 값으로 설정해주면 되는데, 따로 설정을 해주지 않으면 `transparent`가 기본값으로 설정된다.

```CSS
body {
  background-color: #4d9fff;
}

h1 {
  background-color: white;
}

h2 {
  background-color: transparent
}
```





## 15. 그림자

그림자의 위치만 설정해주면 그림자가 나타난다.

```CSS
.div1 {
  background-color: #eeeeee;
  width: 400px;
  height: 300px;
  box-shadow: 40px 10px;
}
```



#### 그림자 색 설정

따로 설정해주지 않으면 그림자는 검정색이다. 만약 다른 색으로 바꾸고 싶으면 `box-shadow`속성에 추가로 색을 써주면 된다.

```CSS
.div1 {
  background-color: #eeeeee;
  width: 400px;
  height: 300px;
  box-shadow: 40px 10px #4d9fff;
}
```



#### 흐림 정도 (blur)

`box-shadow` 속성에서 그림자가 얼마나 흐리게 나올지 설정해줄 수 있다. 가로, 세로 위치 뒤에 추가해주면 되며 기본값은 `0px`이다.

```CSS
.div1 {
  background-color: #eeeeee;
  width: 400px;
  height: 300px;
  box-shadow: 40px 10px 10px #4d9fff;
}
```



### 그림자 크기 (spread)

그림자가 얼마나 퍼질지도 설정할 수 있다. 흐림 값 이후에 써주면 된다.

```CSS
.div1 {
  background-color: #eeeeee;
  width: 400px;
  height: 300px;
  box-shadow: 40px 10px 10px 20px #4d9fff;
}
```





## 16. box-sizing

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Box Model</title>
        <meta charset="utf-8">
        <style>
            /* 모든 요소에 border-box 설정하는 방법
            * {
            	box-sizing: border-box;
            }
            */
            h1 {
                width: 300px;
                height: 200px;
                padding: 35px;
                border: 5px solid red;
                /* box-sizing 속성에 border-box를 써주면 패딩과 테두리가 포함 */
                /* 기본 설정값: content-box */
                box-sizing: border-box;
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





## 17. box-sizing 정리

```CSS
.div1 {
  border: 10px solid red;
  width: 300px;
  height: 200px;
  margin-bottom: 50px;
}

.div2 {
  border: 10px solid red;
  width: 300px;
  height: 200px;
  padding: 40px;
}
```

`.div1`과 `.div2`의 `width`와 `height`를 똑같이 설정해줬는데, 결과물을 보면 크기가 서로 다르다. 그 이유는 `width`와 `height`가 테두리(border)와 패딩(padding)을 뺀 내용물(content)의 크기만 나타내기 때문이다.

따라서 `.div1`의 실제 가로 길이는 테두리까지 포함한 `320px`, 세로 길이는 테두리까지 포함한 `220px`인 것이다. 반면 `.div2`의 실제 가로 길이는 테두리와 패딩까지 포함한 `400px`, 세로 길이는 `300px`이다.

실제 가로, 세로 크기가 `300px`, `200px`이기 위해서는 테두리와 패딩을 고려해서 계산을 해줘야 한다는 불편함이 있다.



### 해결책: box-sizing

다행히 CSS3부터는 `box-sizing` 속성을 사용하면 이 문제를 해결할 수 있다. 따로 설정해주지 않으면 `box-sizing`의 기본값은 `content-box`인데, 이걸 `border-box`로 바꾸면 된다.

```css
.div1 {
  box-sizing: border-box;
  border: 10px solid red;
  width: 300px;
  height: 200px;
  margin-bottom: 50px;
}

.div2 {
  box-sizing: border-box;
  border: 10px solid red;
  width: 300px;
  height: 200px;
  padding: 40px;
}
```

`box-sizing` 속성을 `border-box` 값으로 설정해주면 `width`와 `height`는 테두리와 패딩과 내용물을 모두 포함한 길이가 된다. 따라서 더 이상 귀찮은 계산을 할 필요가 없다.



#### 더 간편하게!

`box-sizing` 속성을 사용하면 너무 편하다 보니, 요즘 많은 개발자들이 **모든** 요소에 `box-sizing: border-box;`를 써주는 추세이다. 이걸 간편하게 한 번에 처리하기 위해서는 모든 요소를 나타내는 `*`에 속성을 써주면 된다.

```CSS
* {
  box-sizing: border-box;
}

.div1 {
  border: 10px solid red;
  width: 300px;
  height: 200px;
  margin-bottom: 50px;
}

.div2 {
  border: 10px solid red;
  width: 300px;
  height: 200px;
  padding: 40px;
}
```





## 20. 배경 이미지

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Background Image</title>
        <meta charset="utf-8">
        <link rel="stylesheet" href="css/styles.css">
    </head>
    
    <body>
        <div class="div1">
            
        </div>
    </body>
</html>
```

```CSS
/* css/styles.css */
.div1 {
    height: 500px;
    border: 5px solid green;
    background-image: url("../images/beach.jpg");
    /* no-repeat에서 repeat로 바꾸면 체크무늬처럼 반복해서 나온다. */
    background-repeat: no-repeat;
    /* 수치 대신 cover를 입력해주면 찌그러지지 않는다. */
    background-size: 100% 500px;
    /* 사진이 잘릴 경우 우선 순위, center center 등 */
    background-position: right bottom;
}
```





## 21. 배경 이미지 정리 노트

배경 이미지에는 여러가지 설정이 가능하다.



### background-repeat

`background-repeat`는 이미지를 반복시킬 것인지 아닐 것인지, 그리고 반복시킨다면 어떤 방식으로 반복시킬 것인지 정해주는 속성이다. 여기에는 우리가 배운 `repeat`, `no-repeat` 외에도 다양한 옵션이 있다.

```CSS
/* 반복하지 않음 */
background-repeat: no-repeat;

/* 가로 방향으로만 반복 */
background-repeat: repeat-x;

/* 세로 방향으로만 반복 */
background-repeat: repeat-y;

/* 가로와 세로 모두 반복 */
background-repeat: repeat;

/* 반복할 수 있는 만큼 반복한 뒤, 남는 공간은 이미지 간의 여백으로 배분 */
background-repeat: space;

/* 반복할 수 있는 만큼 반복한 뒤, 남는 공간은 이미지 확대를 통해 배분 */
background-repeat: round;
```



### background-size

`background-size`는 배경 이미지의 사이즈를 정해주는 속성이다.

```CSS
/* 원래 이미지 사이즈대로 출력 */
background-size: auto;

/* 화면을 꽉 채우면서, 사진 비율을 유지 */
background-size: cover;

/* 가로, 세로 중 먼저 채워지는 쪽에 맞추어서 출력 */
background-size: contain;

/* 픽셀값 지정 (가로: 30px, 세로: 50px로 설정) */
background-size: 30px 50px;

/* 퍼센트값 지정 (가로: 부모 요소 width의 60%, 세로: 부모 요소 height의 70%로 설정) */
background-size: 60% 70%;
```



### background-position

`background-position`은 배경 이미지의 위치를 정해주는 속성이다.

```CSS
/* 단어로 지정해주기 (가로: left, center, right, 세로: top, center, bottom) */
/* 아래와 같은 총 9개의 조합이 가능 */
background-position: left top;
background-position: left center;
background-position: left bottom;
background-position: right top;
background-position: right center;
background-position: right bottom;
background-position: center top;
background-position: center center;
background-position: center bottom;

/* 퍼센트로 지정해주기 (가로: 전체 width의 25% 지점, 세로: 전체 height의 75% 지점 ) */
background-position: 25% 75%;

/* 픽셀로 지정하기 (가로: 가장 왼쪽 가장자리에서부터 오른쪽으로 100px 이동한 지점, 세로: 가장 상단 가장자리에서 아래로 200px 이동한 지점) */
background-position: 100px 200px;
```

