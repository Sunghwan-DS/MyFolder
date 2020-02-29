# href / src/ url

### href / src / url 각각 가장 많이 사용되는 케이스

href : a태그에 홈페이지 등의 주소를 입힐때 사용.

src : img태그에 파일 디렉토리 경로에있는 어떠한 이미지를 지성하여 웹 페이지에 결과물을 출력할때 사용.

url : CSS / html의 style 에서 img태그와 같이 어떠한 파일을 불러올때 사용.



```html
<a href="https://www.codeit.kr">코드잇 바로가기</a>

    <img src="../images/codeit.png" alt="코드잇 이미지"/>
    
    <span style="background:url(../images/codeit.png)"/>
```

```CSS
span{background:url(../images/codeit.png);}
```



### 기억하기 쉬운 정리

1. CSS에서: 항상 "url"
2. HTML에서
   1. link인 경우: "href"
   2. link가 아닌 경우(이미지, 비디오 등등): "src:"