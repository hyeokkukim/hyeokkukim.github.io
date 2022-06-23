---
title:  "[Econ] Endogeneity"
excerpt: "변수속 또다른 변수 - 내생성"
toc: true
toc_sticky: true

categories:
  - Econometrics

use_math: true

---


## <span style="color:#FF9F29;">  1. Endogeneity

Endogeneity(내생성)은 독립변수가 오차항과 상관관계를 가지는 것이다.

예를들어 $$y_1 = x_1' \beta_1 + e$$ 라는 선형회귀 식이 있다고 하자. 이때 $$E(x_1 e) \ne 0$$ 이라면 확률변수 $x_1$ 과 오차항사이에 공분산이 존재한다는 것이다. 즉, 종속변수 $$y_1$$ 에 영향을 주는 우리가 관찰하지 못한 또 다른 확률변수가 존재한다는 것이다. 이때 확률변수 $x_1$ 이 endogeneity가 있다고 한다.



## <span style="color:#FF9F29;">  2. Simultaneous Equations Models

내생성 문제에서 가장 많이 사용되는 Simultaneous Equations Models(SEM)은 수요와 공급 관련 문제이다. $h_s$ 는 노동자들의 연단위 노동시간이고, $w$ 는 시간당 평균 월급이다. 그리고 이 두 변수는 아래와 같이 간단하게 나타날 수 있다.

- Supply equation: $$h_s = \beta_0 + \beta_1w +v$$

(오차항 $v$는 노동공급에 영향일 끼칠 수 있는 기계설비 비용 등이 포함된다.)

이와같은 회귀식을 structural equation이라 한다.

하지만, 앞서 말한것 처럼 오차항 $v$는 우리가 관찰할 수 없지만 오차항에 포함된 변수들이 변화함에 따라 노동공급량도 변화하게 된다. 

그리고 시간당 노동 수요인 $h_d$는 다음과 같이 표현된다.

- Demand equation: $$h_d = \alpha_0 + \alpha_1 w + u$$

(오차항 $u$는 노동수요에 영향을 끼칠 수 있는 땅의 크기 등이 포함된다.)

이와 같은 supply equation과 demand equation은 특정국가 i에서 균형점을 찾게 되는데 이 균형점이 해당 국가에서의 노동 수요와 공급의 최적점이 된다.

- market equilibrium: $$h_{is} = h_{id}=h_i$$

결과적으로 공급식과 수요식을 결합했을때 우리는 한 국가의 노동 수요 공급 식을 이와 같이 나타낼 수 있고,

$$h_i = \beta_o + \beta_1w_i + v_i, h_i = \alpha_0+ \beta_1w_i + u_i$$ 이 두 식이 simultaneous equation model이 되는것이다.

$$\alpha_1 \ne \beta_1$$ 라는 전제하에 두 식을 연립하여 풀게 되면

$$w_i = \frac{\beta_0-\alpha_0}{\alpha_1-\beta_1} + \frac{v_i-u_i}{\alpha_1-\beta_1} ,h_i=\frac{\alpha_0\beta_0-\alpha_0\beta_1}{\alpha_1-\beta_1} + \frac{\alpha_1v_i-\beta_1u_i}{\alpha_1-\beta_1}$$  과 같이 나타낼 수 있고, 이 두식을 reduced formdm으로 나타내면 $$w_i = \pi_0+e_{i1}, h_i=\delta_0+e_{i2}$$ 이와 같이 간소화 가능하다.

또한, 위 두 식은 $$Cov(w_i,u_i)=\frac{Var(u_i)}{\alpha_1-\beta_1}, Cov(w_i,v_i)=\frac{Var(v_i)}{\alpha_1-\beta_1}$$ 을 만족하게 된다.

그러므로 $$\alpha_1 <0,\beta_1>0$$ 을 만족한다면 wage는 노동 수요와 양의 상관관계를 갖고, 노동 공급과 음의 상관관계를 가지게 된다.

이 결과 SEM모델을 OLS 추정을 통한 회귀계수의 추정값은 endogeneity bias를 생성하고, 불편성을 만족하지 못하게 된다.

$$\sqrt{n}(\hat{\alpha_1}-\alpha_1)\rightarrow_p\frac{Cov(w_i,u_i)}{Var(w_i)}$$



## <span style="color:#FF9F29;">  3. Motivation

그렇다면 내생성을 어떻게 제거할 수 있을까?

도구변수를 사용하는 것이다.

다음 회귀식 $$y=\beta_0 + x_1\beta_1 +x_2\beta_2 + \cdots+x_{k-1}\beta_{k-1}+x_k+\beta_k +e$$ 에 대해

확률변수 $x_k$ 는 내생변수(endogenous)이고 나머지 확률변수 $$x_1,\cdots{},x_{k-1}$$ 은 외생변수(exogenous)라고 한다면 내생변수의 내생성을 제거하기 위해 우리는 instrumental variables(도구변수) $z_1$ 로 내생성을 제거한다.

instrumental variables $z_1$는 오차항과 상관관계를 갖지 않는다. 즉, 첫번째 식 입장에서 보면 $z_1$은 외생변수인 셈이다.

그리고 이 내생변수는 도구변수의 영향을 받기 때문에 $$x_k = \gamma_0 + x_1\gamma_1 + \cdots + x_{k-1}\gamma_{k-1}+z_1\theta_1+\gamma_k$$ 와 같이 표현된다. 이때, 내생변수 $x_k$의 도구변수 집합 $$Z=(x_1, \cdots, x_{k-1}, z_1)'$$ 이 된다.

결과적으로 우리는 도구변수들로부터 → 외생변수를 추정 → 추정된 외생변수와 기존 독립변수들로 종속변수 추정의 과정을 거쳐, 외생변수에 내생성을 제거하게 된다.

> 즉, 처음의 회귀식은 $$y = \alpha_0 + x_1\alpha_1 + \cdots + x_{k-1}\alpha_{k-1}+\lambda _1z_1 + v$$ 가 된다.
(단, $$v=e+\beta_k\gamma_k, \alpha_j = \beta_j + \beta_k\gamma_j(j=1,\cdots,k-1) \text{ and } \lambda_1 = \beta_k\theta_1$$ )



## <span style="color:#FF9F29;">  4. Instrumental Variables

해당 내용들을 정리하면

확률변수 샘플 $$(x_i,y_i,z_{i1})$$ 에 대해 선형회귀식 $$y_i = x_i'\beta + e_i$$ 로 표현가능하고, $z_i$는 $$I\times 1$$ 의 도구변수이고, $x_i$는 $$k \times 1$$ 의 독립변수이자 내생성을 가진다면 우리는 독립변수 $x_i$를 외생변수 $x_{1i}$와 내생변수$x_{2i}$로 나누어 추정할 수 있다.

$$x_i = {x_{1i} \choose x_{2i}},$$ where $$x_{1i}$$ is $$ k_1 \times 1, x_{2i}$$ is $$ k_2 \times 1$$ and $$E(x_{1i}e_i)=0,E(x_{2i}e_i) \ne 0$$

그리고 해당 회귀식에서의 도구변수는 $$z_i = {x_{1i} \choose z_{2i}}$$ where $$x_{1i}$$ is $$k_1 \times 1, z_{2i} \text{ is } l_2\times 1$$ and $$x_{1i} = z_{1i}$$ 로 추정하게 된다.



## <span style="color:#FF9F29;">  5. 2SLS

앞선 내용들을 통해 최종적으로 회귀계수를 추정할때는 두가지 케이스로 다르게 추정가능하다.

첫번째로 설명변수의 수와 도구변수의 수가 동일할때$(k=I)$의 추정은 아래와 같다.

$$\hat{\beta}_{IV} = \hat{\Gamma}^{-1}\hat{\lambda} = [(Z'Z)^{-1}Z'X]^{-1}[(Z'Z)^{-1}Z'y] = (Z'X)^{-1}(Z'Z)(Z'Z)^{-1}Z'y=(Z'x)^{-1}Z'y$$

두번째로 설명변수의 수보다 도구변수의 수가 더 많을때는 두번에 걸쳐 추정하게 되는데 이를 two-stage least squares(2SLS)라 한다.

첫번째로 내생변수의 추정값을 계산한다. 

$$\hat{X} = PzX = Z\hat{\Gamma}=Z(Z'Z)^{-1}Z'X$$

그리고 해당 추정값으로 회귀변수를 추정하게 된다.

$$\hat{\beta}_{2SLS}=(X'PzX)^{-1}X'Pzy=(\hat{X}'\hat{X})\hat{X}'y$$

즉, 종속변수 $y_i$, 내생변수(endogenous) $x_{2i}$, 관찰가능한 외생변수(included exogenous) $x_{1i}=z_{1i}$, 숨겨진 외생변수(excluded exogenous) $z_{2i}$, 도구변수(instrumental) $$Z = [z_{1i},z_{2i}]$$ 에 대한 회귀식은 아래와 같다.  

$$
\begin{aligned}
y_1 &= x_{1i}\beta_{1i}+x_{2i}\beta_{2i}+e_i\\

&=x_{1i}\beta_{1i}+z_{1i}\alpha_{1i}+z_{2i}\alpha_{2i}+u_i+e_i\\

&=x_{1i}\beta_{1i}+x_{1i}\alpha_{1i}+z_{2i}\alpha_{2i}+u_i+e_i\\
\end{aligned}
$$
결과적으로 우리는 내생성을 제거할 수 있게되는 것이다.