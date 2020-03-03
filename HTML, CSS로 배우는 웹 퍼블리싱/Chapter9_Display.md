# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 9. Display

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

