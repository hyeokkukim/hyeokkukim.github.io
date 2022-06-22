---
title: "Python 2. 기본기"
categories: 
    - "python"
tags:
    - "python"
    - "파이썬"    
    - "기본 용어"   
last_modified_at: 2020-02-25T23:00:00+09:00
---

미국에서 개인적으로 프로그래밍 기본/전문 용어가 익숙하지 못해서 미팅때 이해를 잘 하지 못하고 여러번 물어봐야하는 경우들이 상당이 있었습니다. 무엇보다 기본이 중요하기 때문에, 기본 용어를 잘 이해하고 머리에 박혀 있어야 다른 co-worker들과 협업할 때 대화가 원할 하게 이루어질 수 있습니다.
<br>

# 표현식(Expressions)과 문장(Statements 혹은 commands)
파이썬에서 어떠한 값(숫자, 수식, 문자열 등)을 만들어 내는 간단한 코드를 표현식(expressions)이라고 부릅니다.
```python
273                   # expression
10 + 20 * 30          # expression
"Hello python"        # expression
+                     # not an expression. + 그것 자체만으로 어떠한 값도 만들어 낼 수 없기 때문에 표현식이 아님.

print("Hello python") #statement

```
<br>

# 키워드(Keyworkds)
특별한 의미가 부여된 단어로, 파이썬이 만들어질 때 이미 사용하겠다고 약속한 단어를 말하며, 파이썬은 대소문자를 구분하는 프로그래밍 언어이기 때문에 대소문자에 따라서 키워드가 될 수 도 있고 안될 수도 있습니다.
```python
True        # keyword
true        # keyword가 아님.
```

만약 사용하고자 하는 단어가 키워드인지 아닌지 확인하고 싶은 경우 아래의 코드로 키워드를 확인할 수 있습니다.
```python
>>> import keyword
>>> print(keyword.kwlist)
['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'] 
```
<br>

# 식별자(identifiers)
변수(variables) 또는 함수의 이름을 명칭합니다. 코드가 길어지거나 다른 사람이 식별자의 이름만으로도 대충 무엇을 의미하는지 알 수 있도록 의미 있는 단어를 사용하는 것이 좋습니다. 그리고 식별자에는 공백을 사용할 수 없기 때문에, 둘 이상의 단어로 조합이 된 경우 **snake_case**나 **CamelCase**를 사용한다. 아래의 규칙을 지키지 않는다고 해서 프로그램 작성에 문제가 되는 것은 아니지만, 파이썬 개발자들이 지켜서 사용하는 규칙이고 식별자만 보았을 때 무엇을 하는 것인지 한번에 이해할 수 있습니다.
- 키워드를 사용하면 안됨.
- 특수 문자는 _만 허용.
- 숫자로 시작하면 안됨.
- 공백을 포함할 수 없음.
- CamelCase는 두 이름 사이를 대문자로 연결되어 있고, 주로 클래스이름에 사용됨.
- snake_case는 두 이름 사이를 _으로 연결되어 있고, 주로 함수(이름 뒤에 ()가 있음) 또는 변수에서 사용됨.
- 파이썬에서는 둘다 사용이 가능하지만, 첫 번째 글자가 소문자라면 무조건 snake_case를, 대문자라면 CamelCase를 사용한다.
```python
print()         # snake_case이고 함수
list()          # snake_case이고 함수
math.pi         # snake_case이고 변수
class Animal    # CamelCase이고 클래스
BeautifulSoup() # CamelCase이고 클래스. 클래스 생성자라고 부르는 특이한 형태의 함수임.
```
<br>

# 주석(Comments)
프로그램의 진행에 영향을 주지 않는 코드로, 프로그램을 설명하기 위해 사용됩니다. 주석 처리를 하고자 하는 부분 앞에 # 기호를 붙여서 주석 처리합니다. 
```python

>>> # 이름을 출력하는 예입니다.
>>> print("Hello, David!")   # 문자열을 출력합니다.
Hello, David!

""""
여러줄 주석은
이렇게 
작성합니다.
"""
```
<br>


> References:
- Introduction to Computation and Programming Using Python (John Guttag / MIT)
- 혼자 공부하는 파이썬 (윤인성 / 한빛미디어)