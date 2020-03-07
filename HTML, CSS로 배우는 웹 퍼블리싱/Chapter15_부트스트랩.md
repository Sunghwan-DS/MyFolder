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

