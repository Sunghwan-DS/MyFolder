# HTML/CSS로 배우는 웹 퍼블리싱

# Chapter1 <수업 소개>

> 이번 강의에 대한 오리엔테이션입니다.

##### 1분 안에 웹사이트 런칭하기

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">  
    <style>
        @import url(https://fonts.googleapis.com/css?family=Raleway:400,200,300,800);
        @import url(https://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css);
        * {
          box-sizing: border-box;
        }
        img {
          width: 50%;
          border-radius: 50%;
          border: 3px solid white;
          transform: scale(1.6);
          position: relative;
          float: right;
          right: -15%;
        }
         h2, p {
          margin: 0;
          text-align: left;
          padding: 10px 0;
          width: 100%;
        }
         h2 {
          font-size: 1.3em;
          font-weight: 300;
          border-bottom: 1px solid rgba(0, 0, 0, 0.2);
        }
        p {
          font-size: 0.9em;
        }
        i {
          line-height: 40px;
          font-size: 24px;
          padding: 5px;
          color: #000000;
        }
        a {
          opacity: 0.3;
          text-decoration: none;
        }
        .container {
          font-family: 'Raleway', Arial, sans-serif;
          position: relative;
          float: left;
          overflow: hidden;
          max-width: 480px;
        }
         .caption {
          padding: 20px 30px 20px 20px;
          position: absolute;
          left: 0;
          width: 50%;
        }
         .bottom {
          padding: 15px 30px;
          font-size: 0.9em;
          color: #ffffff;
          clear: both;
          background: #20638f;
        }
    </style>
</head>
<body>

  <div class="container blue">
    <div class="caption">
      <h2>THE <b>1975</b></h2>
      <p>The 1975 are an English rock band originating from Manchester.</p>
      <div class="icons">
        <a href="https://www.facebook.com/" target="_blank"><i class="ion-social-facebook"></i></a>
        <a href="https://www.instagram.com/" target="_blank"><i class="ion-social-instagram-outline"></i></a>
        <a href="https://twitter.com/" target="_blank"><i class="ion-social-twitter"></i></a>
      </div>
    </div>
    <img src="https://i.imgur.com/JZjQwYS.jpg"/>
    <div class="bottom">Alternative Rock Band</div>
  </div>

</body>
</html>
```



##### index.html

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">  
    <style>
        @import url(https://fonts.googleapis.com/css?family=Raleway:400,200,300,800);
        @import url(https://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css);
        * {
          box-sizing: border-box;
        }
        img {
          width: 50%;
          border-radius: 50%;
          border: 3px solid white;
          transform: scale(1.6);
          position: relative;
          float: right;
          right: -15%;
        }
         h2, p {
          margin: 0;
          text-align: left;
          padding: 10px 0;
          width: 100%;
        }
         h2 {
          font-size: 1.3em;
          font-weight: 300;
          border-bottom: 1px solid rgba(0, 0, 0, 0.2);
        }
        p {
          font-size: 0.9em;
        }
        i {
          line-height: 40px;
          font-size: 24px;
          padding: 5px;
          color: #000000;
        }
        a {
          opacity: 0.3;
          text-decoration: none;
        }
        .container {
          font-family: 'Raleway', Arial, sans-serif;
          position: relative;
          float: left;
          overflow: hidden;
          max-width: 480px;
        }
         .caption {
          padding: 20px 30px 20px 20px;
          position: absolute;
          left: 0;
          width: 50%;
        }
         .bottom {
          padding: 15px 30px;
          font-size: 0.9em;
          color: #ffffff;
          clear: both;
          background: #20638f;
        }
    </style>
</head>
<body>

  <div class="container blue">
    <div class="caption">
      <h2>THE <b>1975</b></h2>
      <p>The 1975 are an English rock band originating from Manchester.</p>
      <div class="icons">
        <a href="https://www.facebook.com/" target="_blank"><i class="ion-social-facebook"></i></a>
        <a href="https://www.instagram.com/" target="_blank"><i class="ion-social-instagram-outline"></i></a>
        <a href="https://twitter.com/" target="_blank"><i class="ion-social-twitter"></i></a>
      </div>
    </div>
    <img src="https://i.imgur.com/JZjQwYS.jpg"/>
    <div class="bottom">Alternative Rock Band</div>
  </div>

</body>
</html>
```



## 수업 소개

HTML / CSS/ JS 를 이용하면 웹사이트를 만들 수 있다.



### HTML

> HyperText Markup Language
>
> 웹사이트에 들어갈 내용을 담당!

```html
<h1>내가 할 일</h1>
<p>
    들어갈 내용
</p>
<ul>
    <li>내용 1</li>
    <li>내용 2</li>
    <li>내용 3</li>
</ul>
```

HTML만 이용할 경우 사이트에 텍스트만 들어가기 때문에 못생겼다.

![image-20200226072930501](C:\Users\전재인\AppData\Roaming\Typora\typora-user-images\image-20200226072930501.png)



## CSS

> Cascading Style Sheets
>
> 웹사이트의 스타일을 담당!

```CSS
h1{
    text-align: center;
}
p{
    color: blue;
}
ul{
    margin-top: 100px;
}
```

![image-20200226072834763](C:\Users\전재인\AppData\Roaming\Typora\typora-user-images\image-20200226072834763.png)



HTML과 CSS가 내용, 스타일을 담당한다면, JS는 인터랙션을 담당한다.