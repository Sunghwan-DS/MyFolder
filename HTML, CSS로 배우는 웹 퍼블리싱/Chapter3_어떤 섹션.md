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

