# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 10. 포지셔닝

> 아이템이 내가 원하는 곳이 아닌 엉뚱한 곳으로 가버린다면? 내가 놓고 싶은 위치에 정확하게 아이템을 놓기 위해서는 포지셔닝을 이해해야 합니다. relative, absolute 등의 포지셔닝을 학습하고 실습합니다.

## 1. relative 포지션

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Position</title>
        <meta charset="utf-8">
        <style>
            div {
                margin: 30px;
                padding: 30px;
            }
            
            .red {
                border: 1px solid red;
            }
            
            .green {
                border: 1px solid green;
            }
            
            .blue {
                border: 1px solid blue;
            }
            
            b {
                font-size: 28px;
                background-color: orange;
                /*
                원래 위치를 기준으로 상대적 위치로 이동
                position: relative;
                top: 30px;
                left: 50px;
                margin을 조작하면 주변 성분들도 영향을 받는다.
                */
            }
        </style>
    </head>
    
    <body>
        <div class="red">
            문단1 문단1 문단1 <b>문단1 문단1</b> 문단1 문단1 문단1
        </div>
        <div class="green">
            문단2 문단2 문단2 문단2 문단2 문단2 문단2 문단2
        </div>
        <div class="blue">
            문단3 문단3 문단3 문단3 문단3 문단3 문단3 문단3
        </div>
    </body>
</html>
```

요소 검사 - Computed 탭 - 설정되있는 속성 (show all) - position : static

static position이면 원래 있어야할 위에치 있다는 뜻.





## 6. fixed 포지션

```html
<!DOCTYPE html>
<html>
    <head>
        <title>Position</title>
        <meta charset="utf-8">
        <style>
            div {
                margin: 30px;
                padding: 30px;
            }
            
            .red {
                border: 1px solid red;
            }
            
            .green {
                border: 1px solid green;
            }
            
            .blue {
                border: 1px solid blue;
            }
            
            b {
                font-size: 28px;
                background-color: orange;
                /* 브라우저를 기준으로 포지셔닝 */
                /* 스크롤해도 똑같은 자리에 위치한다. */
                position: fixed;
                top: 30px;
                left: 50px;
            }
        </style>
    </head>
    
    <body>
        <div class="red">
            문단1 문단1 문단1 <b>문단1 문단1</b> 문단1 문단1 문단1
        </div>
        <div class="green">
            문단2 문단2 문단2 문단2 문단2 문단2 문단2 문단2
        </div>
        <div class="blue">
            문단3 문단3 문단3 문단3 문단3 문단3 문단3 문단3
        </div>
    </body>
</html>
```





## 7. absolute 포지션

포지셔닝이 안 된 요소

- Static

포지셔닝이 된 요소

- Relative - 원래 위치가 기준
- Fixed - 브라우저 윈도우가 기준
- Absolute - 가장 가까운 포지셔닝이 된 조상(Ancestor) 요소가 기준



```html
<!DOCTYPE html>

<html>
    <head>
        <title>Absolute Position</title>
        <meta charset="utf-8">
        <style>
            .red {
                background-color: red;
                width: 500px;
                height: 500px;
            }
            
            .green {
                background-color: green;
                width: 300px;
                height: 300px;
                position: relative;
                top: 40px;
                left: 90px;
            }
            
            .blue {
                background-color: blue;
                width: 100px;
                height: 100px;
                /* 포지셔닝이 된 가장 가까운 조상은 green */
                position: absolute;
                bottom: 40px;
                right: 10px;
            }
        </style>
    </head>
    
    <body>
        <div class="red">
            <div class="green">
                <div class="blue">
                </div>
            </div>
        </div>
    </body>
</html>
```

