# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter 5. 텍스트 스타일링

## 1. 텍스트 색

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Color Example</title>
        <meta charset="utf-8">
        <style>
            h1 {
                color: rgb(97, 249, 107);
                /* color: #61F96B */
            }
        </style>
    </head>
    
    <body>
        <h1>색깔을 바꿔봅시다!</h1>
    </body>
</html>
```

W3Schools  =>  CSS color 140개의 색 명칭

HTML COLOR CODES  =>  RGB 확인 가능, HEX(RGB를 16진법으로 바꾼 값)

HEX는 복사, 붙혀넣기만 하면 되서 편하다.





## 2. 텍스트 색 정리

텍스트의 색을 지정해주기 위해서는 color 속성을 사용하면 되는데 CSS에서 색을 표현하는 방식은 세 가지가 있다.



### 색 이름

CSS에서 정해준 색 이름 중 하나를 쓰는 방법이다. 모든 색이 있지는 않지만, 무려 140개의 색이 대부분 브라우저에서 지원된다.

```CSS
h1{
    color: blue;
}
```



### RGB 값

모든 색은 빨강(Red), 초록(Green), 파랑(Blue)의 조화로 표현할 수 있다. 이 표현 방식이 바로 'RGB' 이다.  [이런 사이트](https://htmlcolorcodes.com/color-picker/)에서 원하는 색을 찾아볼 수 있다.

```CSS
h1 {
    color: rgb(83, 237, 65)
}
```



### HEX 값 (16진법)

HEX 값은 단순히 RGB 값을 16진법으로 표현한 방식이다. `83`는 16진법으로 `53`이고, `237`는 16진법으로 `ED`이고, `65`는 16진법으로 `41`이다. 따라서 `rgb(83, 237, 65)`는 `#53ED41`와 같다.

```CSS
h1 {
    color: #53ED41;
}
```





## 3. 몇 가지 텍스트 스타일링

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Color Example</title>
        <meta charset="utf-8">
        <style>
            p {
                font-size: 32px;
                font-weight: 700;
                /* left, right, center */
                text-align: center;
                /* underline, overline, line-through */
                text-decoration: underline;
            }
            
            a {  /* 링크의 밑줄을 제거할 수 있다. */
                text-decoration: none;
            }     
        </style>
    </head>
    
    <body>
        <p>텍스트를 다양하게 스타일링 해봅시다!</p>
    </body>
</html>
```

font-weight가 지원하는 값은 100, 200, 300 등 100 단위이다. (없는 값은 기본 굵기로 설정)

폰트나 브라우저에 따라 사용 가능한 범위가 각각 다르다.





## 4. 폰트 굵기 설정

폰트 굵기를 설정하기 위해서는 font-weight 속성을 사용하면 됩니다.



### 사용법

사용 가능한 값은 얇은 순서로 `100`, `200`, `300`, `400`, `500`, `600`, `700`, `800`, `900`이다. `100`이 가장 얇고, `900`이 가장 굵다는 뜻.

```CSS
#p1 {
  font-weight: 400;
}

#p2 {
  font-weight: 700;
}

#p3 {
  font-weight: normal;
}

#p4 {
  font-weight: bold;
}
```

`font-weight: normal;`은 `font-weight: 400`과 똑같고, `font-weight: bold;`는 `font-weight: 700`과 똑같다.



### 주의 사항

- `150`, `230`과 같은 값은 사용할 수 없다. 만약 사용한다면 그냥 기본값으로 설정된다.
- 폰트나 브라우저에 따라서 지원하는 폰트 굵기 값이 다르다. 어떤 폰트는 `100`, `400`, `700`만 지원될 수도 있다는 뜻.





## 5. 가운데 정렬이 안 돼요

```html
<!DOCTYPE html>

<html>
    <head>
        <title>가운데 정렬</title>
        <meta charset="utf-8">
        <style>
            h1 {
                text-align: center;
            }
            
            .menu {
                text-align: center;
            }
            
            /* 링크는 딱 단어만큼만 공간을 차지하여 center가 작동하지 않는다. */
            /* 때문에 <div> 태그로 링크를 감싸면 전체 공간을 차지한다 */ 
            a {
                text-align: center;
            }     
        </style>
    </head>
    
    <body>
        <h1>Hello World!</h1>
        <p>Paragraph</p>
        <div class="menu">
            <a href="#">Link</a>
        </div>
    </body>
</html>
```





## 6. 텍스트 정렬

`text-align` 속성을 사용하면 텍스트를 왼쪽, 오른쪽, 또는 가운데로 정렬할 수 있다.

```CSS
#p1 {
  color: red;
  text-align: left;
}

#p2 {
  color: green;
  text-align: right;
}

#p3 {
  color: blue;
  text-align: center;
}
```



### 예제

`<p>`태그나 헤더 태그들 뿐만 아니라 `<div>`태그의 내용물도 정렬을 할 수 있다.

```html
<style>
	.navigation {
        text-align: center;
	}
</style>

<div class="navigation">
  <a href="#">Menu 1</a> <a href="#">Menu 2</a> <a href="#">Menu 3</a>
</div>
```





## 7. 텍스트 꾸미기 (text-decoration)

`text-decoration`을 사용하면 텍스트를 몇 가지 방법으로 꾸밀 수 있습니다.



### Underline

`underline` 값을 사용하면 밑줄이 그어진다.

```CSS
h1 {
  text-decoration: underline;
}
```



### Overline

`overline` 값을 사용하면 글 위에 줄이 그어진다.

```CSS
h1 {
  text-decoration: overline;
}
```



### Line-through

`line-through` 값을 사용하면 줄이 글을 관통한다.

``` CSS
h1 {
  text-decoration: line-through;
}
```



### None

`none` 값을 사용하면 아무 줄도 없으며 이것이 기본 값이다.

```CSS
h1 {
  text-decoration: none;
}
```

##### `<a>`태그와 사용

사실 `text-decoration`을 가장 많이 사용하는 경우는 텍스트를 꾸미기 위해서가 아니라 꾸밈을 없애기 위해서이다. <a> 태그는 기본적으로 밑줄이 그어져 있는데, 이걸 없애기 위해서 text-decoration: none;을 사용한다.





## 8. 폰트 크기

폰트의 크기를 설정하는 방법 2가지

- Absolute (절대적) - px, pt
- Relative (상대적) - em, %



```html
<!DOCTYPE html>

<html>
    <head>
        <title>Styling Text!</title>
        <meta charset="utf-8">
        <style>
            body {
                font-size: 16px;
            }
            
            .div1 {
                font-size: 100%;
            }
            
            .div2 {
                font-size: 200%;
            }
            
            .div3 {
                font-size: 200%;
            }
        </style>
    </head>
    
    <body>
        <div class="div1">
            div1
            <div class="div2">
                div2
                <div class="div3">
                    div3
                </div>
            </div>
        </div>
    </body>
</html>
```

1pt 는 1px 보다 1.33배 정도 크다. 웹사이트를 만들 때는 pt 보다는 px를 많이 이용한다.

% 로 폰트 크기를 설정하는 경우 부모 요소의 폰트 크기를 기준으로 한다.

1em 은 100%, 2em은 200% 와 같은 효과를 보인다.





## 9. line-height

`line-height`를 사용하면 줄간격을 조절할 수 있다. 사실 `line-height` 속성을 '완벽하게' 이해하려면 타이포그래피 지식이 필요하지만 간단하게 설명하면 다음과 같다.

실제 내용이 들어가는 부분은 'content area'(콘텐츠 영역)이다. `font-family`와 `font-size`에 따라서 'content area'가 정해지고, `line-height`는 'content area'에 영향을 주지 않는다.

`line-height`를 통해서는 각 줄이 실질적으로 차지하는 공간을 정해줄 수 있다. 예를 들어서 `99px`로 설정하면 'content area'보다 `40px`이 많기 때문에 위 아래로 `20px`의 공간이 추가로 생긴다.

반대로 `40px`로 설정하면 'content area'보다 `19px`이 적기 때문에 위 아래로 `-9.5px`의 공간이 줄어든다.

### 코드 예시

```CSS
p {
  font-size: 40px;
  color: white;
}

.p1 {
  background-color: red;
  line-height: normal;
}

.p2 {
  background-color: green;
  line-height: 80px;
}

.p3 {
  background-color: blue;
  line-height: 30px;
}
```





## 10. 폰트 설정

폰트는 크게 5종류로 나눌 수 있다.

1. Serif
   - Times New Roman
   - 궁서체
2. Sans-Serif
   - Arial
   - 굴림체
3. Monospace
   - Courier
   - Courier New
4. Cursive
   - Comic Sans MS
   - Monotype Corsiva
5. Fantasy
   - Impact
   - Haettenschweiler



```html
<!DOCTYPE html>

<html>
    <head>
        <title>Trying Out Different Fonts</title>
        <meta charset="utf-8">
        
        <style>
            /* 이미 글씨체가 컴퓨터에 깔려있는 경우 바로 사용 가능 */
            /* Times New Roman이 없는 경우 Times를 사용하고 이도 없다면 다른 serif 폰트를 사용한다. */
            /* 띄어쓰기가 없는 Times의 경우 따움표가 없어도 된다. */
            h1 {
                font-family: "Times New Roman", Times, serif;
            }
        </style>
    </head>
    
    <body>
        <h1>Trying Out Different Fonts!</h1>
    </body>
</html>
```





## 11. 구글 폰트

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Trying Out Different Fonts</title>
        <meta charset="utf-8">
        
        <link href="https://fonts.googleapis.com/css?family=Barrio|Roboto" rel="stylesheet">
        
        <style>
            h1 {
                font-family: 'Roboto', sans-serif;
            }
            
            h2 {
                font-family: 'Rarrio', cursive;
            }
        </style>
    </head>
    
    <body>
        <h1>Trying Out Different Fonts!</h1>
        <h2>How Exciting!</h2>
    </body>
</html>
```

컴퓨터에 글씨체가 깔려있지 않아도 직접 폰트 파일을 제공해주면 사용 가능하다.

`fonts.google.com` 에서 원하는 폰트에 +버튼을 누른다.

그리고 링크 태그를 복사하고 붙여넣는다.

`fonts.google.com/earlyaccess` 에서 한글 폰트도 확인할 수 있다.

ctrl+f 로 한글 글꼴을 찾고 url만 복사하여 가져온다.





## 12. 각 폰트 보여주기

```html
<!DOCTYPE html>
<html>
<head>
    <title>Fonts</title>
    <meta charset="utf-8">
    <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Barrio">
    <style>
        #serif {
            font-family: Serif;
        }
        
        #sans-serif {
            font-family: Sans-serif;
        }
        
        #monospace {
            font-family: Monospace;
        }
        
        #cursive {
            font-family: Cursive;
        }
        
        #fantasy {
            font-family: Fantasy;
        }
        
        #google {
            font-family: 'Barrio';
        }
    </style>
</head>
<body>
  <div id="serif">Serif</div>
  <div id="sans-serif">Sans-Serif</div>
  <div id="monospace">Monospace</div>
  <div id="cursive">Cursive</div>
  <div id="fantasy">Fantasy</div>
  <div id="google">Google</div>
</body>
</html>
```

