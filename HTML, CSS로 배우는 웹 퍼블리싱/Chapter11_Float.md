# HTML / CSS로 배우는 웹 퍼블리싱

# Chapter 11. Float

> float 속성을 잘 활용하면 여러 아이템의 배치를 보기 좋게 만들어줄 수 있습니다. 여러 그리드로 나뉘어진 사이트를 직접 만들어보면서, 보다 복잡한 형태의 사이트를 만들 수 있는 능력을 길러봅시다.

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





## 4. multiple floats

![image-20200304200238610](C:\Users\전재인\Desktop\git\lecture\HTML, CSS로 배우는 웹 퍼블리싱\image-20200304200238610.png)

![image-20200304200324603](C:\Users\전재인\Desktop\git\lecture\HTML, CSS로 배우는 웹 퍼블리싱\image-20200304200324603.png)

![image-20200307070004209](C:\Users\전재인\Desktop\git\lecture\HTML, CSS로 배우는 웹 퍼블리싱\image-20200307070004209.png)

![image-20200307070057479](C:\Users\전재인\Desktop\git\lecture\HTML, CSS로 배우는 웹 퍼블리싱\image-20200307070057479.png)

그리드(Grid) 레이아웃 : 사이트 내용을 정리하는 가로 세로 틀



