# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 2 <HTML/CSS 시작하기>

> 웹 페이지의 최소 단위인 태그를 배우고, 간단한 사이트를 직접 만들어봅시다.

## 2. 기본 HTML 태그 정리

시작 태그				종료 태그

   <태그>	  내용	  </태그>

예)

```html
<title> 코드잇 - 온라인 프로그래킹 스쿨 </title>
```



```html
<!DOCTYPE html>
<!-- 웹브라우저에게 HTML 버전을 알려주는 역할 (이렇게 쓰면 자동으로 html5 사용) -->
<title> My First Website </title>  <!-- 웹사이트의 제목 -->
<h1> My First Page </h1>  <!-- 가장 큰 머리말 (heading 1) -->
<h2> I love HTML! </h2>  <!-- 두 번째로 큰 머리말 (heading 2) -->

<p>  <!-- paragraph - 긴 글을 쓸 때 사용 -->
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
</p>
```

![image-20200311110732077](upload\image-20200311110732077.png)



## 4. `<b>`태그, `<i>`태그 정리

```html
<p>
    이 부분은 <b>굵게</b> 써주세요.  <!-- bold -->
</p>
<p>
    이 부분은 <i>날려서</i> 써주세요.  <!-- italic -->
</p>
```

bold `<b>`와 비슷한 효과로 strong `<strong>` 이 있는데 bold는 글씨만 굵게 만들지만 strong은 실제로 강조 표시를 남겨 프로그램을 통해 이용이 가능하다.

emphasized`<em>` 또한 italic`<i>`와 글씨를 기울이는 점에서는 같지만 emphasized 또한 글씨에 강조 표시를 남겨 기능적으로 이용이 가능하다.



## 5. 한글이 깨져요

```html
<!DOCTYPE html>

<title> My First Website </title>

<h1> My First Page </h1>
<h2> I <i>love</i> HTML! </h2>
<h3> 안녕 세상! </h3>

<p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod <i>tempor</i> incididunt ut labore et dolore magna aliqua.</p>
```

위의 html을 Chrome에서 열면 한글이 제대로 나오지만 Safari에서 열면 한글이 깨져 나온다. 이유는 한글이 제대로 나오기 위해서는 한글을 인식하는 인코딩 방식을 이용해야하나 Safari에서 이를 지원하지 않는 방식으로 보인다.

```html
<!DOCTYPE html>

<title> My First Website </title>
<meta charset="utf-8">
<!-- 한글을 지원하는 대표적인 인코딩 방식 (종료 태그 필요X) -->

<h1> My First Page </h1>
<h2> I <i>love HTML!</i> </h2>
<h3> 안녕 세상! </h3>

<p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod <i>tempor</i> incididunt ut labore et dolore magna aliqua.</p>
```





## 6. CSS 기초

CSS 기본 문법

```css
/* 스타일링 하고 싶은 요소 */
h1 {
    font-size: 64px;	/* 속성과 속성 값 */
    text-align: center;
}
/* h1의 폰트 사이즈를 64px로 설정, h1의 글을 가운데 정렬 */
```





## 7. 기본 CSS 속성 정리

### 폰트 크기

### 텍스트 정렬

### 텍스트 색

### 여백

```html
<!DOCTYPE html>

<title> My First Website </title>
<meta charset="utf-8">

<h1> My First Page </h1>
<h2> I <i>love</i> HTML! </h2>
<h3> 안녕 세상! </h3>

<p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod <i>tempor</i> incididunt ut labore et dolore magna aliqua.</p>

<style>
h1 {
    font-size: 64px;
    text-align: center;
}
    
h3 {
    margin-top: 100px;	/* h3 위에 여백 100px */
}

p i {	/* p태그 안에 있는 i태그 */
	font-size: 48px;
}
</style>
```



```html
<!-- 여기에 html 코드 -->
<h1>Heading 1</h1>
<h2>Heading 2</h2>
<h3>Heading 3</h3>


<style>
/* 여기에 CSS 코드 */
    h1{
        text-align: left;
        color: lime;
        margin-bottom: 80px;
    }
    h2 {
        font-size: 72px;
        text-align: right;
        color: hotpink;
    }
    h3 {
        text-align: center;
        color: blue;
        margin-left: 50px
    }
    
</style>
```





## 8. 텍스트 꾸미기 연습

```html
<!DOCTYPE html>
<!-- 한글이 깨지지 않도록 코드 추가 -->
<meta charset="utf-8">
<title>Codeit</title>

<h1>내 첫 <i>HTML</i> 프로젝트</h1>
<h2>이름: 이윤수</h2>
<h3>이메일: yoonsoo@codeit.kr</h3>
<p>나는 <i>HTML</i>을 좋아한다. 앞으로 이 강의를 듣고, 나만의 <b>웹사이트</b>를 만들어볼 계획이다. 코드잇과 함께라면 무엇이든 가능하리라 믿는다. 아자아자 화이팅!</p>
<p>이번 <i>HTML</i> 수업 뒤에는 무엇이 기다리고 있을까? 설레는 마음으로 이번 과정을 끝낸 후, 다음 <i>JavaScript</i> 수업을 들어 <b>웹사이트</b>를 더 역동적으로 만들어봐야겠다!</p>

<style>
/* 여기에 CSS 코드 추가 */
    h1 {
        text-align: center;
        color: lime;
    }
    h2 {
        text-align: right;
        color: hotpink;
    }
    h3 {
        text-align: right;
        color: blue;
    }
    p i {
        font-size: 64px;
        color: green;
    }
</style>
```





## 10. head, body, html 태그

```html
<!DOCTYPE html>

<html>
    <head>
        <title> My First Website </title>
        <meta charset="utf-8">
        <style>
            h1 {
                font-size: 64px;
                text-align: center;
            }

            h3 {
                margin-top: 100px;	/* h3 위에 여백 100px */
            }

            p i {	/* p태그 안에 있는 i태그 */
                font-size: 48px;
            }
            </style>
    </head>

    <body>
        <h1> My First Page </h1>
        <h2> I <i>love</i> HTML! </h2>
        <h3> 안녕 세상! </h3>

        <p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod <i>tempor</i> incididunt ut labore et dolore magna aliqua.</p>
    </body>
</html>
```

body 태그: 페이지에 나오는 내용을 감싸줌.

head 태그: 제목, CSS, JAVASCRIPT 등 내용 외의 여러 가지가 들어 간다.

html 태그: 태그 사이에 있는 내용이 html이라는 뜻





## 11. 옵셔널 태그, 꼭 써야 할까?

`<html>`, `<head>`, `<body>` 태그 없이도 별 문제없이 작동하였다. 그 이유는 이 세 태그는 사실 필수가 아니라 '옵셔널 태그'이기 때문이다.

위에서는 다음의 세 태그가 정리(organization)의 목적으로 사용되었다. 요소들을 `<head>`와 `<body>`에 묶어주면 html 파일의 구조가 눈에 더 잘 들어온다고 생각하기 때문이다.

하지만 세 옵셔널 태그의 사용을 권장하지 않는 의견들도 있다. 심지어  [구글 HTML/CSS 스타일 가이드](https://google.github.io/styleguide/htmlcssguide.html#Optional_Tags)에서도 옵셔널 태그를 생략하라고 나와 있습니다.

개인적인 작업을 할 때는 직접 결정하면 되고, 팀으로 작업을 할 때는 상의 후 정하면 된다.





## 12. 링크

하이퍼링크  =>  `<a>` 태그

```html
<a href="https://google.com">구글로 가는 링크</a>
```

![image-20200311110818993](upload\image-20200311110818993.png)



```html
<!DOCTYPE html>

<html>
    <head>
        <title> My First Website </title>
        <meta charset="utf-8">
        <style>
            h1 {
                font-size: 64px;
                text-align: center;
            }

            h3 {
                margin-top: 100px;
            }

            p i {
                font-size: 48px;
            }
            </style>
    </head>

    <body>
        <h1> My First Page </h1>
        <h2> I <i>love</i> HTML! </h2>
        <h3> 안녕 세상! </h3>

        <p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod <i>tempor</i> incididunt ut labore et dolore magna aliqua.</p>
        
        <a href="https://google.com"
           target="_blank">구글로 가는 링크</a>  <!-- target="_blank"는 새 탭에서 열게 해줌 -->
    </body>
</html>
```



이 외에도 다음과 같은 방식 가능.

```html
<!-- 하위폴더로 넘어가기 -->
<a href="folder1/page1.html">page 1</a>
<a href="folder1/folder2/page2.html">page 2</a>

<!-- 상위폴더로 넘어가기 -->
<a href="../index.html">index</a>
<a href="../../index.html">index</a>
```





## 13. 이미지

```html
<!DOCTYPE html>

<html>
    <head>
        <title> My First Website </title>
        <meta charset="utf-8">
        <style>
            h1 {
                font-size: 64px;
                text-align: center;
            }

            h3 {
                margin-top: 100px;
            }

            p i {
                font-size: 48px;
            }
            
            /* 이미지 가운데 정렬 */
            img {
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
            </style>
    </head>

    <body>
        <h1> My First Page </h1>
        <h2> I <i>love</i> HTML! </h2>
        <h3> 안녕 세상! </h3>
        
        <!-- 길이를 너비, 높이 중에 하나만 적으면 비율대로 확대, 축소된다. -->
        <img src="https://assets3.thrillist.com/v1/image/1656352/size/tmg-slideshow_l.jpg" width="673" height="300">
        <!-- 내가 가지고 있는 이미지 호출 -->
        <!-- <img src="../images/ice_cream.jpg" width="300"> -->

        <p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod <i>tempor</i> incididunt ut labore et dolore magna aliqua.</p>
        
        <a href="https://google.com"
           target="_blank">구글로 가는 링크</a>
    </body>
</html>
```





## 14. 사이즈 설정

### 픽셀

HTML에서 무언가의 크기를 설정할 때는 기본적으로 '픽셀(px)' 단위를 사용한다.

픽셀은 화면을 구성하는 기본 단위이다.



##### 폰트 크기

폰트 크기도 픽셀로 설정하는 경우가 많은데, 폰트 크기가 24px로 설정되어 있으면 폰트의 세로 길이가 24px이라는 뜻이다.



### 퍼센트

길이를 픽셀 말고 퍼센트(%)로 설정할 수도 있다.

```html
<img src="https://i.imgur.com/CDPKjZJ.jpg" width="100%">
<img src="https://i.imgur.com/CDPKjZJ.jpg" width="50%">
<img src="https://i.imgur.com/CDPKjZJ.jpg" width="25%">
```

이미지의 가로 세로 비율은 가로를 기준으로 정해지기 때문에 세로(height)만의 %는 의미가 없다.





## 15. 프로필 페이지

```html
<!DOCTYPE html>
<html>
    <head>
        <title>유재석</title>
        <meta charset="utf-8">
        <style>
            img {
                display: block;
                margin-left: auto;
                margin-right: auto;
            }
        </style>
    </head>
    
    <body>
        <h1>유재석</h1>
        <img src="이미지 주소">
        <p>대한민국의 코미디언, MC, 가수다. 3사 연예대상과 백상예술대상을 통틀어 총 15회 대상 수상을 한 <b>역대 최다 대상 수상자</b>이며, 지상파 방송 3사와 백상예술대상에서 모두 대상을 수상 이른바 그랜드슬램을 달성한 단 2명의 예능인 중 1명이다. 백상예술대상 TV부문 대상까지 수상하며 이제는 대상을 넘어서 문화훈장까지 넘보는 현역 연예인으로 평가받고 있다. 2004년 처음 설문조사에서 인기 개그맨 1위에 오른 후 처음 <b>국민MC</b>라는 타이틀이 붙기 시작했고 이후 약 16년간 확고부동한 대한민국 대표 방송인, 코미디언으로 인정받고 있다. 또한 <i>까임방지권</i> 소유자이기도 하다.</p>
        <a href="https://namu.wiki/w/%EC%9C%A0%EC%9E%AC%EC%84%9D">유재석 나무위키</a>
    </body>
</html>
```

