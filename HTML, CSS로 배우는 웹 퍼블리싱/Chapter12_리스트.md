# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 12. 리스트

> 여러 항목, 메뉴 등이 나열되는 경우에는 리스트를 활용하여 더 유용하게 페이지를 제작할 수 있습니다. 리스트의 사용법을 배워봅니다.

## 1. 리스트

```html
<!DOCTYPE html>

<html>
    <head>
        <title>리스트</title>
        <meta charset="utf-8">
    </head>
    
    <body>
        <!-- Orderd List (<ol>): 순서가 있는 리스트 -->
        <!-- Unordered List (<ul>): 순서가 없는 리스트 -->
        <ol type="1"> <!-- 기본설정값, 이외에는 "A", "a", "I", "i" 등이 있다. -->
            <!-- List Item (<li>) -->
            <li>집 청소</li>
            <li>영어 단어 외우기</li>
            <li>영화 보기</li>
        </ol>
    </body>
</html>
```





## 2. 리스트 스타일링

```html
<!DOCTYPE html>

<html>
    <head>
        <title>리스트</title>
        <meta charset="utf-8">
    </head>
    
    <body>
        <ol type="1">
            <li>집 청소</li>
            <li>영어 단어 외우기</li>
            <li>영화 보기</li>
        </ol>
    </body>
</html>
```

```CSS
ul {
    padding-left: 0;
    width: 200px;
}

li {
    list-style: none;
    margin-bottom: 10px;
    background-color: #77abff;
    color: white;
    padding: 10px 20px;
}
```

<li>` 태그에 요소 검사를 해보면 display: list-item 이라고 나와있다. display: block과 비슷한데 좌측에 리스트를 나열하는 것이 붙어 있다. list-item이기 때문에 붙어있는 것이므로 display를 block으로 바꾸면 사라진다.