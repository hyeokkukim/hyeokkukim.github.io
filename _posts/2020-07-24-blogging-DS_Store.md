---
title: ".DS_Store 파일 제거"
categories: 
    - "blogging"
tags:
    - "blogging"
    - ".DS_Store"
    - "jekyll"
last_modified_at: 2020-07-24T12:00:00+09:00
---
맥에서 블로그 업데이트 하고 commit하려고 보면 가끔 .DS_Store 파일이 존재하는데, 깃헙에 블로그 업뎃할 때는 불필요할 것 같아서 제거 하는 방법을 설명 하고자 합니다.
<br>
<br>

# DS_Store 정의
.DS_Store (Desktop Services Store)의 약자로 윈도우 시스템에서는 thumb.db 파일과 비슷하지만 애플 mac OS X 시스템이 폴더에 접근할 때 생기는 해당 폴더에 대한 메타데이터를 저장하는 파일을 말하고 해당 디렉토리 크기, 아이콘의 위치 및 폴더의 배경에 관련한 정보들을 얻을 수 있습니다. 보통 맥 OS에서만 생성 및 사용이 되지만, 파일 공유 과정에서 .DS_Store 파일도 함께 공유되는 경우가 있습니다. 그리고 해당 파일은 대부분 쓸모 없는 경우가 많지만 forensic 관점에서 유용하게 사용되는 정보들에는 아래와 같습니다.
- spotlight comment 정보. 
: 맥에서는 파일에 원하는 태그를 걸면, spotlight에서 빠르게 검색이 가능하는데 이때 구조체의 comment 영역에서 얻을 수 있습니다.
- 파일의 time stamp정보.
: 구조체의 modD, moDD 영역에 해당하는 부분인데, 파일의 수정 시간과 관련된 정보를 얻을 수 있습니다.
- 해당 디렉토리의 논리적, 물리적 크기에 대한 정보.
: 구조체의 logS, lg1S, phyS, ph1S에서 얻을 수 있습니다.

<br>
# DS_Store 파일 삭제 방법
맥 OS에서 작업 후 GitHub Desktop을 사용하여 push하게 되면 .DS_Store 파일도 함께 push되는 경우가 있습니다. 개인적으로 필요도 없어서 삭제를 하는 편입니다.
1. Repository에서 현재 존재하는 .DS_Store파일 제거 
```
find . -name .DS_Store -print0 | xargs -0 git rm -f --ignore-unmatch
```
2. .gitignore파일로 .DS_Store 라인 추가
```
(base) sunwookkim@MacBook-Pro thinkreen.github.io % echo .DS_Store >> .gitignore
(base) sunwookkim@MacBook-Pro thinkreen.github.io % git add .gitignore
(base) sunwookkim@MacBook-Pro thinkreen.github.io % git commit -m '.DS_Store banished!'
[master c8fe107] .DS_Store banished!
 2 files changed, 1 insertion(+)
 create mode 100644 .gitignore
 delete mode 100644 test/_sass/.DS_Store
```