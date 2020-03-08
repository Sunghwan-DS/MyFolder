# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 3 <어떤 섹션>

> 클래스와 아이디를 통해 HTML/CSS의 구조를 이해하고, 직접 사이트에 적용해봅시다.

## 1. 클래스 (class)

```html
<!DOCTYPE html>

<html>
    <head>
        <title>id/class example</title>
        <meta charset="utf-8">
        <style>
            .big-blue-text {
                font-size: 64px;
                color: blue;
            }
            
            .centered-text {
                text-align: center;
            }
        </style>
    </head>
    
    <body>
    	<h1 class="centered-text">Heading 1</h1>
    	<h2 class="big-blue-text centered-text">Heading 2</h2>
    
    	<p>첫 번째 문단</p>
    	<p>두 번째 문단</p>
    	<p class="big-blue-text">세 번째 문단</p>
    	<p>네 번째 문단</p>
    </body>
</html>
```

class를 사용하면 여러 요소들에게 같은 스타일을 입힐 수 있고 한 요소에 다양한 스타일을 입힐 수 있다.





## 2. 아이디 (id)

```html
<!DOCTYPE html>

<html>
    <head>
        <title>id/class example</title>
        <meta charset="utf-8">
        <style>
            .big-blue-text {
                font-size: 64px;
                color: blue;
            }
            
            .centered-text {
                text-align: center;
            }
            
            #best-text{
                color: orange;
            }
        </style>
    </head>
    
    <body>
    	<h1 class="centered-text">Heading 1</h1>
    	<h2 class="big-blue-text centered-text">Heading 2</h2>
    
    	<p id="best-text">첫 번째 문단</p>
    	<p>두 번째 문단</p>
    	<p class="big-blue-text">세 번째 문단</p>
    	<p>네 번째 문단</p>
    </body>
</html>
```



### class와 id 의 차이점

```html
<!-- class -->
<p class="big-text">문단 1</p>
<p>문단 2</p>
<p class="big-text">문단 3</p>
<p>문단 4</p>
<!-- 중복 클래스 가능 -->


<!-- id -->
<p id="best-text">문단 1</p>
<p>문단 2</p>
<p id="best-text">문단 3</p>
<p>문단 4</p>
<!-- 중복 아이디 불가능 -->
<!-- 틀린 코드 -->


<!-- class -->
<p class="big blue">문단 1</p>
<p>문단 2</p>
<p>문단 3</p>
<p>문단 4</p>
<!-- 여러 클래스 가능 -->


<!-- id -->
<p id="best first">문단 1</p>
<p>문단 2</p>
<p>문단 3</p>
<p>문단 4</p>
<!-- 아이디 하나만 가능 -->
<!-- 틀린 코드 -->
```

여러 요소를 스타일링 하고 싶으면?  =>  class

한 요소만 스타일링 하고 싶으면?  =>  id





## 3. '클래스(class)'와 '아이디(id)' 정리

HTML 요소에게 '이름'을 주는 방법은 두 가지가 있다.

- 클래스 (class)
- 아이디 (id)



#### 클래스 vs 아이디

1. 같은 클래스 이름을 여러 요소가 가질 수 있지만, 같은 아이디를 여러 요소가 공유할 수 는 없다.
2. 한 요소가 여러 클래스를 가질 수 있지만, 한 요소는 하나의 아이디만 가질 수 있다. ( 단, 한 요소가 클래스도 여러 개 갖고 아이디도 하나 가질 수 있다!)

(미리 배우는 우선 순위: html코드의 태그 속에 직접적으로 스타일을 선언하는 inline style이 가장 우선순위가 높고 id, class, tag 순서대로 우선순위가 결정된다.)





## 8. `<div>` 태그

```html
<!DOCTYPE html>

<html>
    <head>
        <title>My Favorite Movies</title>
        <meta charset="utf-8">
        <style>
            h1 {
                text-align: center;
                margin-top: 75px;
                margin-bottom: 75px;
            }
            
            .movie {
                background-color: #eee;
                border-radius: 20px;
                margin-bottom: 50px;
                padding: 50px;
                width: 500px;
                margin-left: auto;
                margin-right: auto;
            }
            
            .movie .title {
                text-align: center;
            }
            
            .movie .poster {
                display: block;
                margin-left: auto;
                margin-right: auto;
                margin-top: 40px;
                margin-bottom: 40px;
            }
            
            }
        </style>
    </head>
    
    <body>
        <h1>My Favorite Movies</h1>
        
        <div class="movie">
            <h2 class="title">
                Eternal Sunshine of the Spotless Mind
            </h2>
            <img class="poster" src="포스터주소">
                내용1
            </p>
        </div>
        
    	<div class="movie">
            <h2 class="title">
                The Truman Show
            </h2>
            <img class="poster" src="포스터주소">
            <p class="summary">
                내용2
            </p>
        </div>
        
    </body>
</html>
```

묶어주고 싶은 요소를 `<div>` 태그로 감싸준다.





## 9. css 파일 따로 쓰기 / link로 연결

```html
<html>
    <head>
        <title>내 소개</title>
        <meta charset="utf-8">
        <!-- link태그를 이용하여 연결할 css파일 설정 -->
        <link href="test_css.css" rel="stylesheet">
    </head>
    
    <body>
        <h1>코드잇</h1>
        <h2>안녕하세요!</h2>
        
        <img src="Cogi.png">
        
        <a href="work.html">작품</a>
        <a href="hobby.html">취미</a>
    </body>
</html>
```



```CSS
body {
    text-align: center;
}

h2 {
    color: gray;
}

img{
    height: 300px;
    display: block;
    margin-left: auto;
    margin-right: auto;
    margin-top: 50px;
    margin-bottom: 50px;
}
```





## 10. 어떤 방식으로 css를 써야 할까?

```html
<html>
    <head>
        <title>내 소개</title>
        <meta charset="utf-8">
        <link href="test_css.css" rel="stylesheet">
    </head>
    
    <body>
        <!-- style 속성을 쓰고 css 코드를 적어주면 바로 적용 가능 -->
        <h1 style="color: red; font-size: 72px;">코드잇</h1>
        <h2>안녕하세요!</h2>
        
        <img src="Cogi.png">
        
        <a href="work.html">작품</a>
        <a href="hobby.html">취미</a>
    </body>
</html>
```

일반적으로 가장 많이 쓰이는 방법은 외부 css 파일을 만들고 link 태그로 연결해주는 것이다.

`<h>` 태그에 직접 넣어 확인해보고 `<style>` 태그에 넣어 사용하다 마지막에 css 파일에 추가한다.





## 11. 스타일을 적용하는 다양한 방법

### 스타일을 적용하는 방법

HTML 코드에 스타일을 입히는 방법에는 세 가지가 있다.



#### 1. `<style>` 태그

```html
<style>
  h1 {
    color: green;
    text-align: center;
  }

  p {
    font-size: 18px;
  }
</style>

<h1>Hello World!</h1>
<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque sit amet lorem sit amet nunc ornare convallis. Pellentesque ac posuere lectus. In eu ipsum et quam finibus fermentum vitae sit amet magna.</p>
```



#### 2. `style` 속성

```html
<h1 style="color: green; text-align: center;">Hello World!</h1>
<p style="font-size: 18px;">Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque sit amet lorem sit amet nunc ornare convallis. Pellentesque ac posuere lectus. In eu ipsum et quam finibus fermentum vitae sit amet magna.</p>
```



#### 3. 외부 CSS 파일 + `<link>` 태그

```CSS
/* css/styles.css */
h1 {
  color: green;
  text-align: center;
}

p {
  font-size: 18px;
}
```

```html
<link href="css/styles.css" rel="stylesheet">

<h1>Hello World!</h1>
<p>Lorem ipsum dolor sit amet, consectetur adipiscing elit. Quisque sit amet lorem sit amet nunc ornare convallis. Pellentesque ac posuere lectus. In eu ipsum et quam finibus fermentum vitae sit amet magna.</p>
```



### 어떤 방법을 써야 할까?

일반적으로는 외부 CSS 파일에 스타일을 쓰고 HTML 코드에서 `<link>` 태그로 연결해주는 것이 가장 좋은 방식이다. 하지만 조금씩 새로운 스타일을 시도해볼 때에는 간편함을 위해서 `<style>`태그를 쓰는 방법 또는 style 속성에서 테스트를 하고, 나중에 외부 CSS 파일로 옮기는 방법도 있다.

