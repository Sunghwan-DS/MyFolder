# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 13. 쇼핑몰

> 지금까지 배운 내용을 통해 쇼핑몰 사이트를 직접 만들어봅니다.

## 1. SimpleShop

```html
<!DOCTYPE html>
<html>
  <head>
    <title>SimpleShop</title>
    <meta charset="utf-8" />
    <link href="css/styles.css" rel="stylesheet" />
  </head>
    
    
  <body>
      <h1 style="font-size: 50px; font-weight: 100;">SimpleShop</h1>
      <div style="border: 1px solid gray;">
      <div id="navbar">
          <img style="padding-left: 30px; height: 20px;" src="images/logo.png">
          <span id="index"><a href="#">contact</a><a href="#">Shop</a><a href="#">Cart</a><a href="#">Login</a></span>
      </div>
          <img style="width: 100%;" src="images/hero_header.jpg">
          <div id="products">
              <h2 style="margin-bottom: 60px; font-size: 24px; color: #545454;"><b>Our New Products</b></h2>
              <div id="one_line">
                  <div id="one_product">
                  <a href="#"><img src="images/sunglasses.jpg">
                      <h3>Sunglasses</h3>
                      <h3>49,000</h3></a>
                  </div>
                  <div id="one_product">
                  <a href="#"><img src="images/tassel_loafer.jpg">
                      <h3>Tassel Loafer</h3>
                      <h3>89,000</h3></a>
                  </div>
                  <div id="one_product">
                  <a href="#"><img src="images/beige_bag.jpg">
                      <h3>Beige Bag</h3>
                      <h3>69,000</h3></a>
                  </div>
              </div>
              <div id="one_line">
                  <div id="one_product">
                  <a href="#"><img src="images/sneakers.jpg">
                      <h3>Sneakers</h3>
                      <h3>79,000</h3></a>
                  </div>
                  <div id="one_product">
                  <a href="#"><img src="images/slippers.jpg">
                      <h3>Slippers</h3>
                      <h3>29,000</h3></a>
                  </div>
                  <div id="one_product">
                  <a href="#"><img src="images/wrist_watch.jpg">
                      <h3>Wrist Watch</h3>
                      <h3>99,000</h3></a>
                  </div>
              </div>
              <div id="one_line">
                  <div id="one_product">
                  <a href="#"><img src="images/fedora_hat.jpg">
                      <h3>Fedora Hat</h3>
                      <h3>39,000</h3></a>
                  </div>
                  <div id="one_product">
                  <a href="#"><img src="images/classic_loafer.jpg">
                      <h3>Classic Loafer</h3>
                      <h3>99,000</h3></a>
                  </div>
                  <div id="one_product">
                  <a href="#"><img src="images/pink_bag.jpg">
                      <h3>Pink Bag</h3>
                      <h3>79,000</h3></a>
                  </div>
              </div>
              
          </div>
      <div id="footer">
          <img src="images/facebook.png"><img src="images/instagram.png"><img src="images/twitter.png">
      </div>
      </div>
  </body>
</html>

```



```CSS
* {
  box-sizing: border-box;
}

body {
  margin: 0;
  min-width: 992px;
  font-family: "Helvetica";
}



/* navbar */
#navbar {
    padding-top: 20px;
    padding-bottom: 20px;
    position: relative;
}

#index {
    position: absolute;
    right: 0;
    padding-right: 30px;
    
}

#index a {
    margin-left: 30px;
    font-weight: bold;
    text-decoration: none;
    color: #545454;
}
/* hero header */

/* products */
#products {
    margin-top: 60px;
    text-align: center;
    margin-bottom: 120px;
}

#one_line {
    margin-bottom: 80px;
}

#products div img {
    width: 225px;
    height: 225px;
    margin-right: 20px;
}

#one_product {
    display: inline-block;
}

#one_product a {
    color: gray;
    text-decoration: none;
}
/* footer */
#footer {
    text-align: center;
    margin-bottom: 80px;
}

#footer img {
    margin-left: 10px;
    margin-right: 10px;
    height: 20px;
}
```

