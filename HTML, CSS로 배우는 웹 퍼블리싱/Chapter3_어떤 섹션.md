# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 3 <어떤 섹션>

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
            <img class="poster" src="data:image/jpeg;base64,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">
                내용1
            </p>
        </div>
        
    	<div class="movie">
            <h2 class="title">
                The Truman Show
            </h2>
            <img class="poster" src="data:image/jpeg;base64,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">
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

