# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 14. 반응형 웹

> 웹의 장점은 다양한 기기에서 손쉽게 실행된다는 것입니다. 다양한 기기와 환경을 지원하도록 반응형으로 웹을 구성하는 방법을 배워봅니다.

## 1. 반응형 웹

브라우저 사이즈에 맞춰서 레이아웃이 바뀌는 것을 반응형 웹 디자인(responsive web design) 이라 한다.

```html
<!DOCTYPE html>

<html>
    <head>
        <title>반응형 웹</title>
        <meta charset="utf-8">
        <link rel="stylesheet" href="css/styles.css">
    </head>
    
    <body>
        <h1>Hello World!</h1>
        <p>내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용 내용
        </p>
    </body>
</html>
```

```CSS
/* css/styles.css */

h1 {
    font-size: 24px;
}

p {
    font-size: 16px;
}

/* 브라우저의 가로길이가 768px 이상일 때는 다음의 스타일을 입힌다. */
@media (min-width: 768px) {
    h1 {
        font-size: 36px;
    }
    
    p {
        font-size: 24px;
    }
}

@media (min-width: 992px) {
    h1 {
        font-size: 48px;
    }
    
    p {
        font-size: 32px;
    }
}
```

