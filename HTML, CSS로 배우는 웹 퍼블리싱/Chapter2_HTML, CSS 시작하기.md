# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 2 <HTML/CSS 시작하기>

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

![image-20200226074614972](C:\Users\전재인\AppData\Roaming\Typora\typora-user-images\image-20200226074614972.png)

## 4. <b>태그, <i>태그 정리

```html
<p>
    이 부분은 <b>굵게</b> 써주세요.  <!-- bold -->
</p>
<p>
    이 부분은 <i>날려서</i> 써주세요.  <!-- italic -->
</p>
```

bold <b>와 비슷한 효과로 strong <strong> 이 있는데 bold는 글씨만 굵게 만들지만 strong은 실제로 강조 표시를 남겨 프로그램을 통해 이용이 가능하다.

emphasized<em> 또한 italic<i>와 글씨를 기울이는 점에서는 같지만 emphasized 또한 글씨에 강조 표시를 남겨 기능적으로 이용이 가능하다.



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

<html>, <head>, <body> 태그 없이도 별 문제없이 작동하였다. 그 이유는 이 세 태그는 사실 필수가 아니라 '옵셔널 태그'이기 때문이다.

위에서는 다음의 세 태그가 정리(organization)의 목적으로 사용되었다. 요소들을 <head>와 <body>에 묶어주면 html 파일의 구조가 눈에 더 잘 들어온다고 생각하기 때문이다.

하지만 세 옵셔널 태그의 사용을 권장하지 않는 의견들도 있다. 심지어  [구글 HTML/CSS 스타일 가이드](https://google.github.io/styleguide/htmlcssguide.html#Optional_Tags)에서도 옵셔널 태그를 생략하라고 나와 있습니다.

개인적인 작업을 할 때는 직접 결정하면 되고, 팀으로 작업을 할 때는 상의 후 정하면 된다.





## 12. 링크

하이퍼링크  =>  <a> 태그

```html
<a href="https://google.com">구글로 가는 링크</a>
```

![image-20200228093128179](C:\Users\전재인\AppData\Roaming\Typora\typora-user-images\image-20200228093128179.png)



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
        <img src="data:image/jpeg;base64,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">
        <p>대한민국의 코미디언, MC, 가수다. 3사 연예대상과 백상예술대상을 통틀어 총 15회 대상 수상을 한 <b>역대 최다 대상 수상자</b>이며, 지상파 방송 3사와 백상예술대상에서 모두 대상을 수상 이른바 그랜드슬램을 달성한 단 2명의 예능인 중 1명이다. 백상예술대상 TV부문 대상까지 수상하며 이제는 대상을 넘어서 문화훈장까지 넘보는 현역 연예인으로 평가받고 있다. 2004년 처음 설문조사에서 인기 개그맨 1위에 오른 후 처음 <b>국민MC</b>라는 타이틀이 붙기 시작했고 이후 약 16년간 확고부동한 대한민국 대표 방송인, 코미디언으로 인정받고 있다. 또한 <i>까임방지권</i> 소유자이기도 하다.</p>
        <a href="https://namu.wiki/w/%EC%9C%A0%EC%9E%AC%EC%84%9D">유재석 나무위키</a>
    </body>
</html>
```

