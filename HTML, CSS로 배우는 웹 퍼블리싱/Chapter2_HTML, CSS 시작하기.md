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

<p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod tempor <i>incididunt</i> ut labore et dolore magna aliqua.</p>
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

<p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod tempor <i>incididunt</i> ut labore et dolore magna aliqua.</p>
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

### 폰트 크리

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

<p>Lorem ipsum <b>dolor</b> sit amet, consectetur adipiscing elit, sed do eiusmod tempor <i>incididunt</i> ut labore et dolore magna aliqua.</p>

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

