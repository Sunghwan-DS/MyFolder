# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 11. Float

## 1. float

![image-20200304094543978](C:\Users\전재인\AppData\Roaming\Typora\typora-user-images\image-20200304094543978.png)

```html
<!DOCTYPE html>

<html>
    <head>
        <title>Float</title>
        <meta charset="utf-8">
        <style>
            .blue {
                background-color: blue;
                width: 300px;
                /* float는 붕 떠있는 모습이기 때문에 겹쳐보인다 */
                /* inline 요소나 inline-block 요소는 붕 뜬 공간을 갈 수 없다 */
                float: left;
            }
            
            .yellow {
                background-color: yellow;
            }
            
            .orange {
                background-color: orange;
            }
        </style>
    </head>
    
    <body>
        <div class="blue"></div>
        <div class="yellow">Hello World! Let's learn about floats!</div>
        <div class="orange"></div>
    </body>
</html>
```

