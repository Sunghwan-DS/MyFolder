# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 8. CSS 제대로 활용하기

> CSS에는 많은 기능이 숨겨져 있습니다. CSS를 잘 활용하면 더 짧고도 정확한 코드를 작성할 수 있습니다.

## 1. 선택자 정리

CSS에서 스타일링 해줄 요소는 '선택자'로 결정한다.



### 태그 이름

```css
/* 모든 <h1> 태그 */
h1 {
  color: orange;
}
```



### 클래스 / 아이디

```css
* 'important'라는 클래스를 갖고 있는 모든 태그 */
.important {
  color: orange;
}

/* 'favorite'라는 아이디를 갖고 있는 태그 */
#favorite {
  color: blue;
}
```



### 자식 (children)

```css
/* 'div1' 클래스를 갖고 있는 요소의 자식 중 모든 <i> 태그 */
.div1 i {
  color: orange;
}
```



### 직속 자식 (direct children)

```css
/* 'div1' 클래스를 갖고 있는 요소의 직속 자식 중 모든 <i> 태그 */
.div1 > i {
  color: orange;
}
```



### 복수 선택

```css
/* 'two' 클래스를 가지고 있는 태그 모두와 'four' 클래스를 가지고 있는 태그 모두 선택 */
.two, .four {
  color: orange;
}
```



### 여러 조건

```css
/* 'outside' 클래스를 갖고 있으면서 'one' 클래스도 갖고 있는 태그 */
.outside.one {
  color: blue;
}

/* 'inside' 클래스를 갖고 있으면서 'two' 클래스도 갖고 있는 태그 */
.inside.two {
  color: orange;
}
```



### Pseudo-class (가상 클래스)

콜론(`:`)을 사용하면 몇 가지 '가상 클래스'를 선택할 수 있다.



#### n번째 자식

```css
/* .div1의 자식인 <p> 태그 중 3번째 */
.div1 p:nth-child(3) {
  color: blue;
}

/* .div1의 자식인 <p> 태그 중 첫 번째 */
.div1 p:first-child {
  color: red;
}

/* .div1의 자식인 <p> 태그 중 마지막 */
.div1 p:last-child {
  color: green;
}

/* .div1의 자식 중 마지막 자식이 아닌 <p> 태그 */
.div1 p:not(:last-child) {
  font-size: 150%;
}

/* .div1의 자식 중 첫 번째 자식이 아닌 <p> 태그 */
.div1 p:not(:first-child) {
  text-decoration: line-through;
}
```



#### 마우스 오버 (hover)

```css
h1 {
  color: orange;
}

/* 마우스가 <h1> 태그에 올라갔을 때 */
h1:hover {
  color: green;
}
```





## 4. CSS 상속

CSS에는 '상속'이라는 개념이 있다. 말 그대로 부모 요소의 속성들을 자식들한테 넘겨주는 것이다.

```html
<div class="div1">
  <h1>Heading 1</h1>
  <p>Paragraph bla bla bla</p>
</div>
```

```CSS
.div1 {
  color: blue;
}
```

`.div1`의 폰트 색을 blue로 설정해주었고, `<h1>`과 `<p>`에 대해서는 별도의 설정이 없다. 그런데도 `<h1>`과 `<p>` 태그의 폰트 색이 파란색으로 설정된다. 그 이유는 `.div1`의 스타일이 자식들에게 상속되었기 때문이다.



#### 상속되는 속성들

하지만 태그와 속성에 따라 상속이 되지 않는 경우도 많이 있다. 예를 들어서 부모 태그에 설정한 `margin`이 모든 자식들에게도 적용되면 큰일이 날 것이다.



##### 웬만하면 상속되는 몇 가지 속성들

1. color
2. font-family
3. font-size
4. font-weight
5. line-height
6. list-style
7. text-align
8. visibility

이외에도 많지만 위는 자주 사용하는 몇 가지 이다.

위에 있는 속성들도 항상 상속되는 것은 아니다. 대표적인 예로 `<a>` 태그에는 color 속성이 상속되지 않는다. `<a>` 태그가 억지로 상속을 받아오기 위해서는 해당 속성에 inherit 값을 쓰면 된다.

```CSS
.div1 {
  color: green;
}

.div2 {
  color: orange;
}

.div2 a {
  color: inherit;
}
```





## 5. CSS 우선 순위

다양한 선택자를 배워봤습니다. 그런데 여러 선택자가 같은 요소를 가리키면 우선 순위를 어떻게 평가할까?



### 순서

완전 똑같은 선택자가 나중에 또 나오면, 이전에 나온 스타일을 덮어쓰게 된다.

```CSS
h1 {
  color: blue;
  text-align: center;
}

h1 {
  color: green;
}
```



## 명시도 (Specificity)

같은 요소를 가리키지만 선택자가 다르다면, '명시도(specificity)'에 따라 우선 순위가 결정된다.



#### 명시도 계산기

명시도 계산 방법은 다음과 같다.

1. 인라인 스타일이 가장 우선 순위가 높다.
2. 선택자에 id가 많을 수록 우선 순위가 높다.
3. 선택자에 class, attribute, pseudo-class가 많을 수록 우선 순위가 높다.
4. 그 다음은 그냥 요소(또는 가상 요소)가 많은 순서이다.



`<ul>` 태그 안에 `<li>` 태그 안에 `<a id="link">` 가 있다고 가정해보자. `<ul>`과 `<li>`는 나중에 배울 '리스트' 태그이다.

첫 번째 경우에는 일반 요소가 세 개, 가상 클래스가 한 개 있어서 '명시도 점수'가 13이다. 두 번째 경우에는 일반 요소가 두 개, 가상 클래스가 한 개, 그리고 id가 한 개 있어서 112점이다.

따라서 두 선택자에서 겹치는 스타일이 있는 경우, 두 번째 경우인 `ul li:first-child #link` 선택자의 스타일이 적용되는 것이다.

```html
<ul>
  <li><a id="link" href="#">Link 1</a></li>
  <li><a id="link" href="#">Link 1</a></li>
  <li><a id="link" href="#">Link 1</a></li>
  <li><a id="link" href="#">Link 1</a></li>
</ul>
```

```CSS
ul li:first-child #link {
  color: green;
}

ul li:first-child a {
  color: orange;
}
```





## 8. CSS의 다양한 단위들

CSS에는 `px`, `rem`, `em`, `%` 등 여러 단위가 있다. 폰트 크기 뿐만 아니라 `padding`, `margin`, `width` 등 다양한 속성들에 이 단위들을 사용할 수 있다.



### px

`px`는 절대적인 값이다. 다른 요소의 값에 영향을 받지 않는다.

```CSS
html {
  font-size: 20px;
}

.container {
  padding-top: 40px;
  background-color: lime;
}
```



### rem

`rem`은 상대적인 값이다. 하지만 오직 `<html>` 태그의 `font-size`에만 영향을 받는다.

`2rem`은 `<html>` 태그의 `font-size`의 2배 크기이다.

```CSS
html {
  font-size: 20px;
}

.container {
  padding-top: 2rem; /* html의 font-size * 2 = 40px */
  background-color: lime;
}
```



### em

`em`도 `rem`과 마찬가지로 상대적인 값이다. `em`은 자기 자신의 `font-size`를 기준으로 한다.

`2em`은 자기 자신의 `font-size`의 2배 크기이다. 자기 자신의 `font-size`를 따로 정해주지 않을 경우, 상위 요소에서 상속받은 값을 기준으로 한다.

```CSS
html {
  font-size: 20px;
}

.container {
  /* 자동으로 html의 font-size 20px을 상속받음 */
  padding-top: 2em; /* 자신의 font-size * 2 = 40px */
  background-color: lime;
}
```



만약 자기 자신에게 정해진 `font-size`가 있다면 그 값을 기준으로 em이 결정된다.

```CSS
html {
  font-size: 20px;
}

.container {
  font-size: 40px;
  padding-top: 2em; /* 자신의 font-size * 2 = 80px */
  background-color: lime;
}
```



### 퍼센트 (%)

`%` 역시 상대적인 값이다. `%`는 어느 항목에서 쓰이느냐에 따라 다른 기준이 적용된다.

예를 들어 `font-size`에서 `%`가 쓰일 경우, 상위 요소의 `font-size`에 곱해주는 방식으로 계산한다.

```CSS
.container {
  font-size: 20px;
  background-color: lime;
}

.text {
  font-size: 180%; /* 상위 요소인 container의 font-size * 1.8 = 36px */
  background-color: skyblue;
  margin: 0;
}
```



`%`가 `margin`이나 `padding`의 단위로 사용될 경우, 상위 요소의 `width`를 기준으로 계산된다.

```CSS
.container {
  width: 200px;
  background-color: lime;
}

.text {
  padding-left: 30%; /* 상위 요소의 width * 0.3 = 60px */
}
```



재미있는 점은 `margin-top`이나 `padding-bottom` 등 세로(상하) 속성를 조절할 때에도 상위 요소의 `height`가 아닌 `width`를 기준으로 계산된다는 것이다.

```CSS
.container {
  width: 200px;
  background-color: lime;
}

.text {
  padding-top: 30%; /* 상위 요소의 width * 0.3 = 60px */
}
```



### 참고

더 자세히 알아보고 싶은 내용은 아래 링크를 참고.

https://webdesign.tutsplus.com/ko/tutorials/comprehensive-guide-when-to-use-em-vs-rem--cms-23984