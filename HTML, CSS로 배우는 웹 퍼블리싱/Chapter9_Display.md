# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 9. Display

> 웹 페이지의 모든 요소에는 display 속성이 있으며, 이 속성을 중심으로 페이지가 구성됩니다. 각각의 상황에 적합한 display 속성은 무엇인지 배워봅시다.

## 1. display

```html
<!DOCTYPE html>

<html>
    <head>
        <title>display</title>
        <meta charset="utf-8">
        <style>
            body {
                font-size: 38px;
            }
            
            i {
                display: block;
            }
        </style>
    </head>
    
    <body>
        Hello my <i>name</i> is young!
        <!-- <i>를 <div> 로 바꾸면 줄이 넘어간다 -->
        <!-- i 태그의 display를 style 태그에서 수정하면 줄이 넘어간다. -->
    </body>
</html>
```



### display 속성

inline, block, inline-block, list-item, table, flex, none, ...

모든 html 요소는 이 중 딱 한 가지 속성을 갖는다.

- inline display
  - 다른 요소들과 같은 줄에 머무르려고 하는 성향
  - 가로 길이는 필요한 만큼만 차지하는 성향
  - 예) `<span>`, `<b>`, `<img>`
- block display
  - 새로운 줄에 가려고 하는 성향
  - 가로 길이를 최대한 많이 차지하려고 하는 성향
  - 예) `<div>`, `<h1>`, `<p>`





## 2. display 정리

### display의 종류

모든 요소는 딱 한 개의 display 값을 갖고 있다. 가질 수 있는 display의 종류는

1. inline
2. block
3. inline-block
4. flex
5. list-item
6. none

등 여러 가지가 있는데, 대부분의 요소들은 inline과 block 중 한 가지이다.



#### inline display

inline 요소들은 다른 요소들과 같은 줄에 머무르려고 하는 성향과, 필요한 만큼의 가로 길이만 차지하는 성향이 있다.

다음 요소들은 기본 display 값이 inline이다.

1. `<span>`
2. `<a>`
3. `<b>`
4. `<i>`
5. `<img>`
6. `<botton>`



```css
i {
  background-color: green;
}
```

`<i>` 태그는 기본적으로 inline이기 때문에 앞, 뒤의 텍스트와 같은 줄에 머무르고 있고, 가로 길이는 필요한 만큼만 차지하고 있다.



#### block display

block 요소들은 다른 요소들과 독단적인 줄에 가려고 하는 성향과, 최대한 많은 가로 길이를 차지하는 성향이 있다.

다음 요소들은 기본 display 값이 block이다.

1. `<div>`
2. `<h1>`, `<h2>`, `<h3>`, `<h4>`, `<h5>`, `<h6>`
3. `<p>`
4. `<nav>`
5. `<ul>`
6. `<li>`



```css
div {
  background-color: green;
}
```

`<div>` 태그는 기본적으로 block이기 때문에 새로운 줄에 가버린다. 그리고 가로 길이는 최대한 많이, 100%를 차지하고 있다.



### display 바꾸기

모든 요소는 기본적으로 정해진 display 값이 있는데, CSS를 통해서 이를 바꿀 수 있다.



#### inline 요소를 block으로 바꾸기

```css
i {
  display: block; /* <i> 태그를 block으로 바꾸기 */
  background-color: green;
}
```



#### block 요소를 inline으로 바꾸기

```css
div {
  display: inline; /* <div> 태그를 inline으로 바꾸기 */
}

.div1 {
  background-color: green;
}

.div2 {
  background-color: blue;
}
```





## 3. inline-block

```html
<!DOCTYPE html>

<html>
    <head>
        <title>display</title>
        <meta charset="utf-8">
        <style>
            body {
                font-size: 38px;
            }
            
            i {
                /* inline display는 길이 개념이 없다.
                width: 300px;
                height: 300px;
                */
                display: inline-block;
                width: 300px;
                height: 300px;
            }
        </style>
    </head>
    
    <body>
        Hello my <i>name</i> is young!
    </body>
</html>
```





## 4. inline-block 정리

Block 요소에게는 가로 길이와 세로 길이를 직접 설정해줄 수 있지만, inline 요소는 자동으로 설정이 된다. Inline 요소에게는 가로, 세로 길이의 개념이 없는 것이다.

만약 inline 요소처럼 다른 요소들과 같은 줄에 머무르면서 block 요소처럼 가로, 세로 길이도 설정해주고 싶으면 어떻게 해야 할까? 바로 그 둘을 섞어놓은 `inline-block`을 사용하면 된다.



### inline-block

```CSS
i {
  display: inline-block;
  width: 200px;
  height: 200px;
  background-color: green;
}
```





## 5. `<img>` 태그의 비밀

`<img>` 태그는 사실 대체 요소(replaced element)라고 하는 특별한 요소이다. (가로 길이 설정 가능)

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Layout</title>
        <style>
            body {
                margin: 0;
            }
            
            /* 이미지를 엄청 큰 글자처럼 취급 */
            img {
                vertical-align: middle;
            }
            
            .container{
                text-align: center;
            }
        </style>
    </head>
    
    <body>
        <div class="container">
            <img="Cogi.png" height="100">
        </div>
    </body>
</html>
```





## 6. 다양한 링크

```html
<a href="https://google.com" target="_blank">
    <!-- 구글로 가는 링크 -->
	<img src="Cogi.png" width="200">
</a>
```





## 9. Baseline

![image-20200305185657185](C:\Users\전재인\Desktop\git\lecture\HTML, CSS로 배우는 웹 퍼블리싱\image-20200305185657185.png)

![image-20200305185819870](C:\Users\전재인\Desktop\git\lecture\HTML, CSS로 배우는 웹 퍼블리싱\image-20200305185819870.png)



```html
<!DOCTYPE html>

<html>
    <head>
        <title>Baseline</title>
        <meta charset="utf-8">
        <link rel="stylesheet" href="css/styles.css">
    </head>
    
    <body>
        <div class="container">
            <span class="alex">alex</span>
            <div class="inline-block">
                <h1>ben</h1>
                <h2>chris</h2>
            </div>
        </div>
    </body>
</html>
```

```CSS
/* css/styles.css */

.container {
    background-color: orange;
}

.alex {
    font-size: 90px;
    background-color: pink;
}

/* inline-block의 경우에는 div의 마지막 줄의 baseline이 baseline으로 설정 */
.inline-block {
    display: inline-block;
    background-color: blue;
    color: white;
    width: 100px;
    height: 150px;
    /* overflow가 visible이 아니면 박스 밑이 baseline으로 설정된다. */
    /* overflow: scroll; */
}
```





## 10. vertical-align pt. 1

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Baseline</title>
        <meta charset="utf-8">
        <link rel="stylesheet" href="css/styles.css">
    </head>
    
    <body>
        <div class="container">
            <span class="alex">alex</span>
            <span class="ben">ben</span>
            <span class="chris">chris</span>
        </div>
    </body>
</html>
```

```CSS
/* css/styles.css */

.container {
    background-color: orange;
}

.alex {
    font-size: 90px;
    background-color: pink;
    /* vertical-align: top; */
}

.ben {
    font-size: 30px;
    background-color: lime;
    /* 초록 박스의 위가 이 줄에서 가장 높은 요소의 위에 맞춰진다. */
    vertical-align: top;
}

.chris {
    font-size: 60px;
    background-color: yellow;
}
```

![image-20200305200704298](C:\Users\전재인\Desktop\git\lecture\HTML, CSS로 배우는 웹 퍼블리싱\image-20200305200704298.png)





## 11. vertical-align pt. 2

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Baseline</title>
        <meta charset="utf-8">
        <link rel="stylesheet" href="css/styles.css">
    </head>
    
    <body>
        <div class="container">
            <span class="alex">alex</span>
            <span class="ben">ben</span>
            <span class="chris">chris</span>
        </div>
    </body>
</html>
```

```CSS
/* css/styles.css */

.container {
    background-color: orange;
}

.alex {
    font-size: 90px;
    background-color: pink;
    /* vertical-align: middle; */
}

.ben {
    font-size: 30px;
    background-color: lime;
    /* 부모태그의 baseline에 중간이 맞춰진다. */
    vertical-align: middle;
}

.chris {
    font-size: 60px;
    background-color: yellow;
}
```





## 12. 세로 가운데 정렬 꿀팁

### 가로 가운데 정렬

#### inline 요소

`inline` 또는 `inline-block` 요소면 부모 태그에 `text-align: center;`를 써주면 된다.

```CSS
.container {
  text-align: center;
  background-color: lime;
}
```



#### block 요소

`block` 요소면 `margin-left: auto;`, `margin-right: auto;`를 써주면 된다.

```CSS
.block-element {
  width: 100px;
  height: 50px;
  margin-left: auto;
  margin-right: auto;
  background-color: lime;
}
```



### 세로 가운데 정렬

사실 CSS에서 모든 걸 한 번에 딱 가운데 정렬을 시키는 방법이 없기 때문에, 지금까지 배운 다양한 지식을 섞어서 해야 한다.

다음은 몇 가지 방법이다.



#### 가짜 요소 더하기

`vertical-align: middle;`을 하면 해결될까요? 우선 `vertical-align` 속성은 인라인 또는 인라인 블록 요소에 적용되기 때문에 `.info`를 인라인 블록으로 바꾸겠습니다. 그리고 `vertical-align: middle;`을 설정해주면...?

```html
<div class="container">
  <div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

```CSS
.container {
  width: 300px;
  height: 400px;
  background-color: gray;
  text-align: center;
}

.info {
  background-color: lime;
  display: inline-block;
  vertical-align: middle;
}
```

`vertical-align: middle;`은 요소의 가운데를 부모 요소의 소문자 'x'의 가운데와 맞춥니다. 확인해봅시다.

```html
<div class="container">
  x
  <div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

`.info` 요소를 완전 가운데로 오게 하려면 우선 소문자 'x'가 가운데로 와야 합니다. 방법이 하나 있습니다. 세로 길이가 `100%`인 요소를 만들고, 그 요소에도 `vertical-align: middle;`을 하는 거죠!

```html
<div class="container">
  x
  <div class="helper"></div>
  <div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

```CSS
.container {
  width: 300px;
  height: 400px;
  background-color: gray;
  text-align: center;
}

.helper {
  display: inline-block;
  height: 100%;
  vertical-align: middle;
  
  /* 설명을 위해서 */
  width: 10px;
  background-color: red;
}

.info {
  background-color: lime;
  display: inline-block;
  vertical-align: middle;
}
```

이제 거의 다 되었습니다. 여기서 소문자 'x'를 지우고, `.helper` 요소의 가로 길이를 없애면 되겠죠?

```html
<div class="container">
  <div class="helper"></div>
  <div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

```CSS
.container {
  width: 300px;
  height: 400px;
  background-color: gray;
  text-align: center;
}

.helper {
  display: inline-block;
  height: 100%;
  vertical-align: middle;
}

.info {
  background-color: lime;
  display: inline-block;
  vertical-align: middle;
}
```

근데 아직도 문제가 조금 있습니다. `.info`의 가로 길이가 `100%`라면 어떻게 되는지 봅시다.

```html
<div class="container">
  <div class="helper"></div>
  <div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

```css
.container {
  width: 300px;
  height: 400px;
  background-color: gray;
  text-align: center;
}

.helper {
  display: inline-block;
  height: 100%;
  vertical-align: middle;
}

.info {
  background-color: lime;
  display: inline-block;
  vertical-align: middle;
  width: 100%;
}
```

갑자기 이상한 곳에 위치되네요. 사실 `.helper` 와 `.info` 요소 사이에 띄어쓰기가 한 칸 있어서, 가로 길이 `100%`인 `.info` 요소는 자리 부족으로 다음 줄로 가버립니다!

이 문제를 해결하기 위해서는 두 가지 방법이 있습니다.

우선 **띄어쓰기를 없애는 방법:**

```html
<div class="container">
  <!-- 스페이스 없애기 -->
  <div class="helper"></div><div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

```CSS
.container {
  width: 300px;
  height: 400px;
  background-color: gray;
  text-align: center;
}

.helper {
  display: inline-block;
  height: 100%;
  vertical-align: middle;
}

.info {
  background-color: lime;
  display: inline-block;
  vertical-align: middle;
  width: 100%;
}
```



**띄어쓰기 공간 만큼의 마이너스 여백을 주는 방법:**

```html
<div class="container">
  <div class="helper"></div>
  <div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

```CSS
.container {
  width: 300px;
  height: 400px;
  background-color: gray;
  text-align: center;
}

.helper {
  display: inline-block;
  height: 100%;
  vertical-align: middle;
}

.info {
  background-color: lime;
  display: inline-block;
  vertical-align: middle;
  width: 100%;

  /* 이 경우 띄어쓰기는 5~7px 정도였습니다! */
  margin-left: -7px;
}
```



**주의 사항:**

어떤 요소에 `height: 100%;`를 설정하기 위해서는 부모의 `height`가 설정되어 있어야 합니다. 위 경우에는 `.helper`의 부모인 `.container`에 `height`가 설정되어 있었기 때문에 가능했던 것이죠.

### `line-height`로 해결

`.info`를 인라인 블록으로 설정해주면, `line-height` 속성을 활용해볼 수도 있습니다. 부모인 `.container`에 `height`와 동일한 `line-height`를 줘보세요.

`line-height` 속성은 자식들에게 상속되기 때문에 `.info`에는 `line-height: normal;`을 꼭 써주셔야 합니다!

```html
<!DOCTYPE html>
<div class="container">
  x
  <div class="info">
    <h1>Hello!</h1>
    <p>My name is young.</p>
  </div>
</div>
```

```CSS
.container {
  width: 300px;
  height: 400px;
  background-color: gray;
  text-align: center;
  line-height: 400px;
}

.info {
  background-color: lime;
  display: inline-block;
  line-height: normal;
  vertical-align: middle;
}
```

### 다른 방식?

위의 방법들 말고도 세로 가운데 정렬을 하는 다양한 방식들이 있습니다. 포지셔닝을 이용할 수도 있고, 최근에 나온 [flexbox](https://www.w3schools.com/css/css3_flexbox.asp)를 이용할 수도 있습니다. 위의 방식으로는 해결되지 않는 상황들도 있을 수 있기 때문에, 다양한 방식들을 연구하는 걸 추천드립니다!