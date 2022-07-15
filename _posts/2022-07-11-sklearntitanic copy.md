---
title:  "[ML for Python] Sklearn을 활용한 타이타닉 생종률 예측"
excerpt: "GMM"
toc: true
toc_sticky: true

categories:
  - Machine Learning

use_math: true

---

## 1. 패키지 import 및 데이터 load

```python
import numpy as np
import pandas as np
df = pd.read_csv('./titanic_train.csv')
df.head()
```
![png](/assets/images/스크린샷_2022-07-07_오후_3.38.16.png){: .align-center}

 해당 파일은 총 12개의 column으로 되어 있다.

Passengerid: 탑승자 데이터 일련번호

Survived: 생존여부(0:사망, 1: 생존)

Pclass: 선실 등급(1: 일등석, 2: 이등석, 3: 삼등석)

Name: 탑승자 이름

Sex: 성별(0: 남성, 1: 여성)

Age: 탑승자 나이

SibSp: 동행자(형제자매, 배우자) 인원수

Parch: 동행자(부모, 자식) 인원수

Ticket: 티켓번호

Fare: 요금

Cabin: 선실 번호

Embarked: 중간 정착 항구(C: Cherbourg, Q: Queenstown, S: Southampton)

## 2. 전처리

우선, 결측치를 확인해보면 Age, Cabin, Embarked에 각각 결측치가 있는것이 확인된다.

```python
df.isnull().sum()
```

![png](/assets/images/스크린샷_2022-07-07_오후_3.46.20.png){: .align-center}

해당 결측치를 처리하기 위해 age는 평균값, cabin은 N, embarked는 N을 넣어준다.

그리고 Cabin의 경우 선실명을 세부구분 지을 필요가 없기 때문에 앞자리만 추출해준다.

```python
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)

df['Cabin'] = df['Cabin'].str[:1]
```

연령대별 분류를 위해 아래와 같이 신규 컬럼 ‘age_cat’을 생성한다.

```python
def get_category(age):
    cat = ''
    if age <= 1: cat = 'unknow'
    elif age <= 5: cat = 'baby'
    elif age <= 12: cat = 'child'
    elif age <= 18: cat = 'teenager'
    elif age <= 25: cat = 'student'
    elif age <= 35: cat = 'young adult'
    elif age <= 60: cat = 'adult'
    else : cat = 'elderly'

    return cat

df['age_cat'] = df['Age'].apply(lambda x: get_category(x))
df.head()
```
![png](/assets/images/스크린샷 2022-07-07 오후 3.56.18.png){: .align-center}

Null 값 처리와 불필요한 피쳐를 제거하고, 문자열 카테고리형 변수들을 변환하기 위해 LabelEncoder 클래스를 이용한다.

```python
#Null 처리
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Cabin'].fillna('N', inplace=True)
df['Embarked'].fillna('N', inplace=True)
df['Fare'].fillna(0, inplace=True)

#불필요한 피처 제거
df.drop(['PassengerId','Name','Ticket'],axis=1,inplace=True)

#레이블링
from sklearn.preprocessing import LabelEncoder
features = ['Cabin','Sex','Embarked']
for feature in features:
    le = LabelEncoder()
    le = le.fit(df[feature])
    df[feature] = le.transform(df[feature])

df.head()
```
![png](/assets/images/스크린샷 2022-07-11 오전 11.25.57.png){: .align-center}

## 3. 모델링

종속변수 survived와 독립변수 Pclass, Sex, Age, SibSp, Parch, Fare, Cabin, Embarked, age_cat을 변수로 설정하고 train set와 test set를 나눈다.

```python
y = df['Survived']
x = df.drop('Survived',axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=11)
```

### 3.1 의사결정나무

의사결정나무를 sklearn의 DecisionTreeClassifier로 시행하고 학습의 accuracy 측정 결과 0.7877이 나왔다.

```python
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(random_state=11)
dt_clf.fit(x_train,y_train)
dt_pred = dt_clf.predict(x_test)
print('DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test,dt_pred)))
```

### 3.2 랜덤포레스트

랜덤포레스트를 sklearn의 RandomForesetCalssifier로 시행하고 학습의 accuracy 측정결과 0.8547이 나왔다.

```python
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state=11)
rf_clf.fit(x_train, y_train)
rf_pred = rf_clf.predict(x_test)
print('RandomForestClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test,rf_pred)))
```

### 3.3 로지스틱회귀

로지스틱 회귀분석을 sklearn의 LogisticRegression으로 시행하고 학습의 accuracy 측정결과 0.8547이 나왔다.

```python
from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression(random_state=11)
lr_clf.fit(x_train, y_train)
lr_pred = lr_clf.predict(x_test)
print('RandomForestClassifier 정확도: {0:.4f}'.format(accuracy_score(y_test,rf_pred)))
```

## 4. 최적 하이퍼 파라미터

GridSearchCV를 이용해 의사결정나무의 최적의 하이퍼 파라미터를 찾고 예측 성능을 측정한다.

CV는 5개의 폴드 세트를 지정하고 하이퍼 파라미터는 max_dept, min_samples_split, min_samples_leaf를 변경하며 성능을 측정한다.

```python
from sklearn.model_selection import GridSearchCV

parameters = {'max_depth':[2,3,5,10],'min_samples_split':[2,3,5], 'min_samples_leaf':[1,5,8]}

grid_dclf = GridSearchCV(dt_clf, param_grid=parameters, scoring='accuracy', cv=5)
grid_dclf.fit(x_train, y_train)

print('GridSearchCV 최적 하이퍼 파라미터:', grid_dclf.best_params_)
print('GridSearchCV 최고 정확도: {0:.4f}'.format(grid_dclf.best_score_))
best_dclf = grid_dclf.best_estimator_
```
![png](/assets/images/스크린샷 2022-07-11 오후 12.40.30.png){: .align-center}

그리고 각 최고의 정확도를 가진 하이퍼 파라미터를 적용했을시의 accuracy는 0.8715로 최초의 0.7877보다 성능이 향상된 것을 확인할 수 있었다.

```python
dpredictions = best_dclf.predict(x_test)
accuracy = accuracy_score(y_test, dpredictions)
print('테스트 세트에서의 DecisionTreeClassifier 정확도: {0:.4f}'.format(accuracy))
```