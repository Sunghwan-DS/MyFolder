# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 10. 포지셔닝

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