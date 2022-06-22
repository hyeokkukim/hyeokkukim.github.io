---
title: "Jupyter Notebook Extensions 주피터 노트북 확장 프로그램 사용하기"
categories: 
    - "jupyter"
tags:
    - "jupyter"
    - "주피터 노트북"
    - "확장 프로그램"
    - "extension"       
last_modified_at: 2020-07-26T23:00:00+09:00
---

Jupyter Notebook을 사용하면서 유용하고 많이 사용하실 만한 `Variable Inspector`나 'Table of Contents'같은 확장 프로그램들에 대하여 소개하고자 합니다.
<br>
<br>

# Install jupyter_contrib_nbextensions 확장 프로그램 설치하기
## pip로 설치하실 경우
```
pip install jupyter_contrib_nbextensions    # 라이브러리 설치
jupyter contrib nbextension install --user  # jupyter notebook에서 사용할 수 있도록 등록
```
## conda로 설치하실 경우
```
conda install -c conda-forge jupyter_contrib_nbextensions   # conda-forge가 제공하는 extension설치
```
<br>

# Use extensions
만약 주피터 노트북이 실행되고 있다면 창을 모두 닫은 뒤에 다시 실행시킵니다.
```
jupyter notebook
```
그 후, [nbextensions](http://localhost:8888/nbextensions) 메뉴로 갑니다.
```
http://localhost:8888/nbextensions
```
정상 설치후 위의 링크로 가면 다음과 같은 웹페이지가 보입니다. 필요하신 확장 프로그램을 선택하고 enable을 누르면 적용이 됩니다.
만약 제일 상단에 `disable configuration for nbextensions`메뉴에 체크박스가 체크 되어있다면 언체크하신 후에, 원하시는 확장 프로그램을 선택하시면 됩니다.
![jupyter extensions](/assets/images/jupyter-extensions.png)