---
title: "Python 라이브러리/패키지/모듈/함수/클래스"
categories: 
    - "python"
tags:
    - "python"
    - "파이썬"
    - "library"
    - "package"
    - "module"        
    - "function"
    - "class"
last_modified_at: 2020-06-30T23:00:00+09:00
---

파이썬을 처음 시작했을 때 라이브러리/패키지, 모듈, 함수/클래스 간의 개념이 머릿속에 정확하게 자리 잡히지 않아서 고생한 적이 있었습니다. 그래서 그 개념을 간략하게 설명하고 다음 포스팅에서 각각의 사용법에 대하여 설명하겠습니다.


![FunctionModuleClass](/assets/images/FunctionModuleClass.png)

# 라이브러리(Library)
라이브러리와 패키지는 둘다 다른 프로그램에서 포함하여 사용할 수 있는 일종의 **코드 모음**이라고 생각하면 쉽습니다. 
- 여러 모듈과 패키지를 묶어 라이브러리라고 합니다.
- 파이썬을 설치할 때 본적으로 설치되는 라이브러리를 **표준라이브러리(Python Standard Library)**라고 합니다. (예: time, sys, os, math, random, urllib 등)
- python.org가 아닌 외부 3rd party에서 개발한 모듈과 패키지를 묶어서 **외부 라이브러리(Third Party Library)**라고 합니다. (예: requests, scrapy, webbrowser 등)
- 3rd party를 통해 개발된 모듈, 패키지가 표준라이브러리보다 더 우수하거나 사용하기 쉬운 경우도 있습니다. (예: Numpy, Scipy, requests)



# 패키지(Package)
일반적으로 패키지와 라이브러리 사이의 경계가 조금 모호한데, 사용중인 프로그래밍 언어마다 그 정의가 조금씩 달라질 수 있습니다. 파이썬에서는 패키지는 패키지 관리자를 사용하여 설치하기 위한 **준비된 라이브러리**입니다. 가끔은 라이브러리를 배포하는 **메커니즘**을 말하기도 합니다.
- 특정 기능과 관련된 여러 모듈들을 하나의 상위 폴더에 넣어 놓은 것을 패키지라고 합니다.
- 패키지 안에 여러가지 하위폴더 가 더 존재할 수 있습니다.
- 파이썬 3.3이후 부터는 `__init__.py`가 없어도 패키지로 인식하지만, 하위 버전 호환을 위해 `__init__.py` 파일을 생성하여 패키지를 표현해주는 것이 안전하다.
- `.`을 이용하여 파이썬 모듈을 계층적으로 관리할 수 있도록 하는 것을 말합니다.
- A.B라고 하면 A는 패키지, B는 모듈이 됩니다. (예: sound.echo --> sound 패키지에 있는 echo모듈)

```
#참고: https://wikidocs.net/1418
#가상으로 game 패키지가 있다고 할 때의 structure 예

game/
    __init__.py
    sound/ 
        __init__.py 
        echo.py 
        wav.py 
    graphic/ 
        __init__.py
        screen.py
        render.py
    play/ 
        __init__.py
        run.py
        test.py

```



# 모듈(Module)
파이썬에서는 모듈의 기능을 활용하여 코드를 분리하고 공유합니다. 이때 여러 기능들(함수, 변수, 클래스 등)을 따로 구현하여 파이썬 파일(.py)을 **모듈(module)**이라고 합니다. 가끔 **script**라고도 부릅니다.
- 파이썬 코드 가장 위에 `import`로 불러오는 것이 모듈입니다. (예: `import math`)


# 함수(Function)
함수는 하나의 기능성을 가진 코드의 집합을 말합니다. 그리고 함수를 사용하는 것을 흔희 **함수를 호출한다**라고 표현하고, 호출할 때 괄호 `( )`내부에 **매개변수**를 넣어 최종적으로 **리턴값**을 부릅니다.

```
def 함수 이름():
    문장
```
- 내장함수(Built-in function): 외부 모듈과는 달리 import를 하지 않습니다. (예: enumerate(), str() )
    ```
    >>> str(3)
    '3'
    >>> str('hi'.upper())
    'HI'
    ```
- 외장함수: import를 통하여 불러드린 외부 모듈 안에 있는 함수를 말합니다. (예: random() )
    ```
    >>> import random
    >>> data = [1, 2, 3, 4]
    >>> random.shuffle(data)
    >>> data
    [4, 1, 3, 2]
    ```

# 클래스(Class)
python을 포함한 Java, Javascript, C#, C++, C등 제가 다룰 줄 아는 언어가 많지 않지만, 이들 중에서 C언어를 제외한 모든 프로그래밍 언어는 **클래스(class)**를 기반으로 **객체(object)**를 만들고, 그러한 객체를 우선으로 생각해서 프로그래밍을 한다는 의미로 **객체 지향 프로그래밍 언어(Object Oriented Programming Language)**라고 합니다. 
- 쉽게 설명한다면 클래스는 붕어빵 틀이라고 보시면 됩니다. 여기서 붕어빵 틀(클래스)를 기반으로 만들어진 객체를 붕어빵(인스턴트)라고 부릅니다.
- 클래스는 객체를 위한 설계도이고, 이러한 클래스를 기반으로 만들어진 객체를 인스턴스라고 합니다.


위에서 간략하게 설명드린 라이브러리, 패키지, 모듈, 함수, 클래스에 대하여 다음 포스팅에서 조금 더 깊게 설명하겠습니다.
 

> References:
- Introduction to Computation and Programming Using Python (John Guttag / MIT)
- 혼자 공부하는 파이썬 (윤인성 / 한빛미디어)