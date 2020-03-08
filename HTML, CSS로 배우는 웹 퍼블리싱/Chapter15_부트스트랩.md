# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 15. 부트스트랩

> 부트스트랩은 사이트를 더 쉽고 간편하게 만들 수 있게 해주는 프레임워크입니다. 부트스트랩을 통해 짧은 코드로 예쁜 디자인을 만들어봅시다. 또한, 다른 프레임워크를 내 코드에 활용할 수 있는 능력도 길러봅시다.

## 1. 새로운 기술을 배우는 법

마지막 챕터입니다.
지금까지 HTML의 많은 기술을 배웠습니다.

하지만 프로그래밍은 빠르게 변합니다.
그리고 웹은 더 빠르게 변합니다.

React, Angular, Vue.js 등 다양한 기술이 등장하고, 매년 주목받는 기술이 달라집니다.

앞으로도 HTML과 CSS도 계속 버전이 업데이트될 것이고, 그 변화를 따라가지 못하면 좋은 개발자가 될 수 없습니다.

**여러분이 좋은 개발자가 되기 위해서는 기술 문서와 커뮤니티를 통해 스스로 새로운 기술을 배울 수 있어야 합니다.**

이번 챕터는 부가적인 프로젝트입니다.
반드시 해야 하는 내용은 아니지만, 지금까지의 내용을 넘어 더 발전하고자 하시는 분들을 위해 마련된 프로젝트입니다.
다른 과제처럼 모든 내용을 알려드리지 않고, 기본적인 내용 몇 가지만 알려드릴 것입니다.
그래서 앞의 내용을 배우셨다고 해서, 결코 쉽게 성공할 수 없는 과제입니다.

제시된 개발 문서와 국내외 커뮤니티를 통해 직접 필요한 내용을 찾으셔야 합니다.

쉽지 않겠지만, 이를 통해 실제로 부딪혀야 할 걸림돌을 미리 확인하고, 스스로 배우는 능력을 기를 수 있을 것입니다.

부가적인 프로젝트이기 때문에 해설도 열어 두었습니다.

직접 한번 시도해보세요!





## 2. 부트스트랩 소개

twitter의 개발자가 오픈 소스로 제공해주는 html, css, javascript 프레임워크이다.

Get start 버튼 - 부트스트랩을 시작하는 방법

웹 상에서 부트스트랩 css파일 받아오기

- 링크 copy - css파일에 연결하는 link태그 위에 복사 붙혀넣기.

웹 상에서 필요한 부트스트랩

- 바디태그 끝에 스크립트 3개 복사 붙혀넣기

프레임워크 = 미리 만들어진 코드





## 3. 부트스트랩 그리드

제가 부트스트랩을 사용할 때 가장 마음에 드는 점은 간편한 그리드 시스템입니다.



### 기본 구성원

부트스트랩 그리드 시스템에는 세 가지 구성원이 있습니다:

1. 컨테이너 (container)
2. 행 (row)
3. 열 (column)



### 기본 규칙

부트스트랩 사이트에 자세히 설명되어 있지만 많은 분들이 무시하는 몇 가지 규칙입니다:

1. 행(`<div class="row">`)은 꼭 컨테이너(`<div class="container">`) 안에 넣어주세요.
2. 열(`<div class="col">`)은 꼭 행(`<div class="row">`) 안에 넣어주세요. 오직 열만 행의 직속 자식이 될 수 있습니다.
3. 콘텐츠(우리가 그리드에 넣고 싶은 내용)는 꼭 열(`<div class="col">`) 안에 넣어주세요.

이 규칙들만 지켜도 예상치 못한 레이아웃이 나오지는 않을 것입니다!



### 기본 사용법

구성원들과 규칙을 알았으면 이제 사용법을 알아봅시다.

부트스트랩 그리드에는 한 줄에 기본적으로 12칸의 열(column)이 있다고 생각하시면 됩니다. 예를 들어서 한 줄을 정확히 3등분하고 싶으면 네 칸을 차지하는 열 세 개를 쓰면 되는 거죠. 네 칸을 사용하는 열은 `<div class="col-4">`입니다.

아래의 코드에서는 다양한 방식으로 12칸을 나누어보았습니다.

```html
<head>
  <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0-beta/css/bootstrap.min.css" integrity="sha384-/Y6pD6FV/Vv2HJnA6t+vslU6fwYXjCFtcEpHbNJ0lyAFsXTsjBbfaDjzALeQsN6M" crossorigin="anonymous">
</head>

<body>
  <div class="container">
    <div class="row">
      <!-- 정확히 3등분 -->
      <div class="col-4 first">first</div>
      <div class="col-4 second">second</div>
      <div class="col-4 third">third</div>
    </div>

    <div class="row">
      <!-- 정확히 2등분 -->
      <div class="col-6 first">first</div>
      <div class="col-6 second">second</div>
    </div>

    <div class="row">
      <!-- 1대 5 비율 -->
      <div class="col-2 first">first</div>
      <div class="col-10 second">second</div>
    </div>

    <div class="row">
      <!-- 1대 2대 1 비율 -->
      <div class="col-3 first">first</div>
      <div class="col-6 second">second</div>
      <div class="col-3 third">third</div>
    </div>
  </div>
</body>
```

```CSS
.container {
  text-align: center;
}

.first {
  background-color: yellow;
}

.second {
  background-color: lime;
}

.third {
  background-color: orange;
}
```



#### 12칸을 넘어가면?

만약 한 행에 12칸이 넘는 열이 들어간다면, 새로운 줄로 넘어가게 됩니다.



#### Why 12?

부트스트랩을 만든 분들은 왜 하필 `12`라는 숫자로 정했을까요?

`12`는 상당히 많은 숫자들(`1`, `2`, `3`, `4`, `6`, `12`)로 나누어지기 때문에 굉장히 유연합니다!

예를 들어서 8칸으로 나누고 싶더라도 `12`라는 숫자의 유연함 덕분에 쉽게 할 수 있습니다. `col-6`를 두 개 쓰면 2등분 할 수 있고, 그 안에서 또 `col-3`로 4등분을 하면 8칸이 생기겠죠?

이런식으로 열을 또 여러 열로 나누는 것을 '중첩(nesting)'한다고 부릅니다. 중첩을 하기 위해서는 우선 열(`<div class="col">`) 안에 새로운 행(`<div class="row">`)을 쓰셔야 합니다. 예제를 통해 살펴보세요:





## 4. 부트스트랩 반응형 그리드

부트스트랩의 그리드 시스템은 반응형 웹 디자인을 할 때 가장 빛을 발합니다.

다음은 부트스트랩에서 정해둔 구간들입니다.

1. Extra Small (< 576px): 모바일
2. Small (≥ 576px): 모바일
3. Medium (≥ 768px): 타블릿
4. Large (≥ 992px): 데스크탑
5. Extra Large (≥ 1200px): 와이드 데스크탑



### 컨테이너 (container)

기본적으로 컨테이너는 가운데 정렬이 되어 있고, 그리드의 행들을 감싸주는 역할을 합니다(행들은 열들을 감싸주고 있고요!). 컨테이너의 종류는 두 가지인데요.

1. `<div class="container">`: 구간별로 그리드에 고정된 width를 설정해줍니다.
2. `<div class="container-fluid">`: 그리드는 항상 width: 100%; 입니다.



#### `<div class="container">`

만약 구간별로 그리드에 고정된 가로값을 설정해주고 싶으면 `"container"` 클래스를 사용하세요. 구간별로 그리드가 고정되어 있으면 레이아웃이 더 예상 가능합니다. 따라서 저는 개인적으로 `"container"` 클래스를 사용하는 것을 선호하고, 디자이너에게 이렇게 구간별로 고정되는 방식으로 만들기를 부탁합니다!



`"container"` 클래스를 사용하면 아래의 CSS 코드가 적용됩니다.

```CSS
.container {
  width: 100%; /* extra small */
  padding-right: 15px;
  padding-left: 15px;
  margin-right: auto;
  margin-left: auto;
}

/* small */
@media (min-width: 576px) {
  .container {
    max-width: 540px;
  }
}

/* medium */
@media (min-width: 768px) {
  .container {
    max-width: 720px;
  }
}

/* large */
@media (min-width: 992px) {
  .container {
    max-width: 960px;
  }
}

/* extra large */
@media (min-width: 1200px) {
  .container {
    max-width: 1140px;
  }
}
```



#### `<div class="container-fluid">`

저는 많은 경우에 `"continer"` 클래스를 선호하지만, 상황에 따라 그리드가 항상 100%의 가로 길이를 갖는 것이 좋을 때가 있습니다. 그럴 때는 `"container-fluid"` 클래스를 사용하면 됩니다.



`"container-fluid"` 클래스를 사용하면 아래의 CSS 코드가 적용됩니다.

```CSS
.container-fluid {
  width: 100%;
  padding-right: 15px;
  padding-left: 15px;
  margin-right: auto;
  margin-left: auto;
}
```



### 열 (column)

반응형 구간별로 (총 12칸 중) 열이 차지하는 칸의 개수도 다르게 할 수 있습니다.

예시를 몇 가지 봅시다.



#### 예시 1 (구간별로 모두 설정되어 있는 경우)

```html
<div class="col-12 col-sm-6 col-md-4 col-lg-3 col-xl-2"></div>
```

1. Extra Small (< 576px): 12칸을 모두 차지
2. Small (≥ 576px): 6칸 차지
3. Medium (≥ 768px): 4칸 차지
4. Large (≥ 992px): 3칸 차지
5. Extra Large (≥ 1200px): 2칸 차지



#### 예시 2 (특정 구간만 설정되어 있는 경우)

아래와 같이 특정 구간에만 열 수가 설정되어 있는 경우도 있습니다. 그렇다면 그 구간부터 새로운 설정이 있는 상위 구간까지는 같은 칸 수를 차지합니다.

```html
<div class="col-12 col-lg-3"></div>
```

1. Extra Small (< 576px): 12칸을 모두 차지
2. Small (≥ 576px): 12칸 차지
3. Medium (≥ 768px): 12칸 차지
4. Large (≥ 992px): 3칸 차지
5. Extra Large (≥ 1200px): 3칸 차지

```html
<div class="col-6"></div>
```

1. Extra Small (< 576px): 6칸 차지
2. Small (≥ 576px): 6칸 차지
3. Medium (≥ 768px): 6칸 차지
4. Large (≥ 992px): 6칸 차지
5. Extra Large (≥ 1200px): 6칸 차지