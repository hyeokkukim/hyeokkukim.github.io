---
title: "minimal-mistakes font 사이즈 변경"
categories: 
    - "blogging"
tags:
    - "blogging"
    - "minimal-mistakes"
    - "jekyll"
    - "폰트 사이즈"
    - "font size"
last_modified_at: 2020-07-23T12:00:00+09:00
---
minimal-mistakes 테마로 블로그를 만들었을 때, 웹 브라우저 화면 사이즈에 따라 폰트 사이즈가 변경이 됩니다. 하지만 개인 적으로 flexible box 레이아웃을 사용하되 폰트 사이즈는 고정으로 하는 것을 더 선호합니다. 그래서 간단하게 폰트 사이즈를 globally 변경하는 방법을 확인 하겠습니다. 
<br>
<br>

# Adjusting font size
해당 요소의 css파일에서 `$type-size-x` 속성을 각각 바꿀수도 있지만 손이 많이 가는 관계로, 폰트 사이즈를 globally 변경하기 위한 쉬운 방법 두 가지에 대해 말씀드리려고 합니다.

## Update main.css
_reset.scss파일이 크고 실수로 다른 부분을 수정해서 overriding하게 되는 리스크가 있습니다. 리스크를 피하고자 따로 폰트 사이즈만 오버라이딩 하도록 [main.scss](https://github.com/thinkreen/thinkreen.github.io/blob/master/assets/css/main.scss) 파일에 새 css를 추가하는 방법을 추천하고, 방법은 아래와 같습니다.
```css
@import "minimal-mistakes";

html {
font-size: 16px; // change to whatever

@include breakpoint($medium) {
    font-size: 18px; // change to whatever
}

@include breakpoint($large) {
    font-size: 18px; // change to whatever
}

@include breakpoint($x-large) {
    font-size: 18px; // change to whatever
}
}
```
![font-size-Screenshot1](/assets/images/font-size-Screenshot1.png)


## Update _reset.scss
[_sass/minimal-mistakes/_reset.scss](https://github.com/thinkreen/thinkreen.github.io/blob/master/_sass/minimal-mistakes/_reset.scss)에 있는 font-size값을 원하는 값으로 수정해 줍니다. 저는 18px가 보기 좋아서 $medium/$large/$x-large 스크린 사이즈에서 모두 18로 변경하였습니다.

![font-size-Screenshot2](/assets/images/font-size-Screenshot2.png)

