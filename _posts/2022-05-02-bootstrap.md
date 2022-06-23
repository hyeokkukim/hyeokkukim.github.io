---
title:  "[Econ] Bootstrap"
excerpt: "bootstrap으로 모집단의 분포 추정"
toc: true
toc_sticky: true

categories:
  - Econometrics

use_math: true

---


### 1. Asymptotic Distribution Tehory

모집단의 parameter distribution $T_n((y_1,x_1),\cdots{},(y_n,x_n),F)$에 대해 $T_n$의 CDF는 다음과 같다.

$$Gn(u,F)=Pr(T_n\leq u|F)$$ 

이때 우리는 모집단의 분포 F를 모르기 때문에 asymptotic하게 분포를 추정할 수 있다.

즉, $G(u,F) = lim_{n \rightarrow \infty}G_n(u,F)$ 모집단의 분포 F를 몰라도 표본 n이 충분히 크면 모집단의 분포를 추정할 수 있다.

만약 추정값이 asymptotically 정규분포를 따른다면 asymptotic distribution은 평균과 분산에 의존하게 된다. 하지만 이 평균과 분산은 찾기 매우 어렵기 때문에 bootstrap을 사용하게 된다.

Definition of the Bootstrap

부트스트랩은 어떤 모집단의 분포의 통계량 추정이 어려울때 사용한다. 부트스트랩과 asymptotic approximations의 차이는 bootstrap은 알 수 없는 $F$대신 임의의 $F_n$을 통해 추정하는 것이고, asymptotic approximation은 $F_n$을 $F$에 근사하는 것이다. 이렇게 우리는 $F_n$을 통해 추정된 부트스트랩의 분포를 $G^*_n=G_n(u,F_n)$ 이라고 할 수 있다.

부트스트랩 과정은 다음과 같다.

1. 분포 $F_n$ 으로부터 $$( y_i^*,x_i^* )$$ 을 샘플링 한다.
2. 1에서 샘플링한 부트스트랩 샘플들로 부터 구하고자 하는 통계량 $$T_n^* = T_n((y_1^*,x_1^*),\cdots{},(y_n^*,x_n^*),F_n)$$ 을 구한다.

nonparametric bootstrap: The Empirical Distribution Function

nonparametric bootstrap의 경우 $F_n$으로 누적분포함수인 EDF를 활용한다.

$F(y,x)=Pr(y_i\leq y,x_i\leq x)=E[1(y_i \leq y)\cdot (x_i\leq x)]$ 라고 할때 $F$ 의 분포는 누적분포함수로 나타내졌다. 이때의 추정에 사용되는 분포 $F_n(y,x)=\frac{1}{n}\Sigma_{i=1}^n1(y_i\leq y)1(x_i \leq x)$ 로 나타내고 이 분포를 Empirical Distribution Funcion인 EDF라고 한다. (단, $F_n$ 은 $F$ 의 nonparametric estimate이다.)

그리고 Weak Law of a Large Numbers에 의해 $F_n(y,x)\rightarrow_pF(y,x)$ 로 수렴하게 되고,

Central Limim Theorem에 의해 $\sqrt{n}(F_n(y,x)-F(y,x))\rightarrow_dN(0,F(y,x)(1-F(y,x)))$ 가 된다.

EDF는 bootstrap sample pair의 확률을 $1/n$이 되게 해주는데 말인 즉슨, $Pr(y_i^* \leq y,x_i^* \leq x)=F_n(y,x)$ 가 된다. 그리고, 해당 부트스트랩 샘플의 moments of function을 구하면 다음과 같다.

$$
\begin{aligned}
E[h(y_i^*,x_i^*)]&=\int h(y,x)dF_n(y,x)\\
&=\Sigma_{i=1}^nh(y_i,x_i)Pr(y_i^*=y_i,x_i^*=x_i)\\
&=\frac{1}{n}\Sigma_{i=1}^nh(y_i,x_i)
\end{aligned}
$$

---

### 2. (non) Parametric Bootstrap procedure

1. $F_n$으로부터 $1/n$의 확률로 random vector ( $y_i^*,x_i^*$ ) n개를 추출한다.
2. 추출된 bootstrap sample로부터 부트스트랩 통계량 $T_n^*$을 계산한다.
3. 1,2번을 B번 반복한다.
4. 각각 추출된 $$T_{nb}^*,b=1,\cdots{},B$$ 로부터 $$T^*_n < u$$ 에 해당하는 empirical probability를 계산한다.

모집단의 분포를 아는 경우 parametric이라고 한다. 이때의 Bootstrap procedure는  $F(\hat{\theta}_n)\rightarrow F(\theta)$ 를 만족하는 $\theta$ 에 대해 bootstrap에 $F_n$ 대신 $F(\hat{\theta}_n)$ 을 사용하면 parametric bootstrap이 된다. 복원추출 샘플링을 했던 EDF경우와 달리 이 경우에는 동일한 샘플사이즈 $n$을 추출하게 된다.

---

### 3. Bootstrap Estimation of Bias

asymptotic estimator와는 다르게 bootstrapdms non-consistency를 전제하기 때문에 bias가 존재한다. $\hat{\theta}$의 bias를 $\tau_n=E(\hat{\theta}-\theta_0)$라고 하면, $T_n(\theta)=\hat{\theta}-\theta$의 bias는 $\tau_n=E(T_n(\theta_0))$ 가 된다.

이때 bootstrap을 통해 bias를 조정이 가능한데 bootstrap sample $$\hat{\theta}^*=\hat{\theta}((y_1^*,x_1^*),\cdots{} ,(y_n^*,x_n^*))$ 와 $T_n^*=\hat{\theta}^*-\theta_n=\hat{\theta}^*-\hat{\theta}$ 에 대하여 bootstrap estimate의 bias $\tau_n=E(T^*_n)$$ 이 된다.

그리고 bias의 추정값은 $$\hat{\tau}_n^*=\frac{1}{B}\Sigma_{b=1}^B T_{nb}^*=\frac{1}{B}\Sigma_{b=1}^B\hat{\theta}^*-\hat{\theta}=\bar{\hat{\theta}^*}-\hat{\theta}$$  

만약 $\hat{\theta}$ 에 bias가 존자하다면 bias를 조정해야 하고, 조정된 bias를 bias-correceted estimator라 한다.

1. $\hat{\theta}<\theta_0$ 라면 $$\bar{\hat{\theta}^*}<\hat{\theta}$$ 이므로 bias-corrected estimator는 $$\tilde{\theta}^*=\hat{\theta}-\hat{\tau}_n^*=\hat{\theta}-(-(\bar{\hat{\theta}^*}-\hat{\theta}))=\hat{\theta}+(\bar{\hat{\theta}^*}-\hat{\theta})=2\hat{\theta}-\bar{\hat{\theta}^*}$$ 이 된다.
2. $\hat{\theta} > \theta_0$ 라면 $$\bar{\hat{\theta}^*}> \hat{\theta}$$ 이므로 bias-corrected estimator는 $$\tilde{\theta}^*=\hat{\theta}-\hat{\tau}_n^*=\hat{\theta}-(\bar{\hat{\theta}^*}-\hat{\theta})=2\hat{\theta}-\bar{\hat{\theta}^*}$$ 이 된다.

---

### 4. Bootstrap Estimation of Variance

$T_n=\hat{\theta}$ 일때 $Var(\hat{\theta})=V_n=E(T_n-E(T_n))^2$ 이고,

$$T_n^*=\hat{\theta}^* $$ 일때 $$Var(\hat{\theta}^*)=V_n^*=E(T_n^*-E(T_n^*))^2$$ 이 된다.

이때 variance의 추정값은 $$\hat{V}_n^*=\frac{1}{B}\Sigma_{b=1}^B(\hat{\theta}_b^* -\bar{\hat{\theta}^*})^2$$ , 

bootstrap의 standard error는 $$s^*(\hat{\theta})=\sqrt{\hat{V}_n^*}$$ 가 된다.

---

### 5. Percentile Interval

bootstrap에서의 confidence interval은 bootstrap을 통해 나온 통계량의 quantile function으로 구한다.

$q^*_n(\alpha)=q_n(\alpha,F_n)$ 일때, $q^*_n(\alpha)$ 은 bootstrap의 quantile 값이 된다.

$T_n=\hat{\theta}$ 가 추정하고자 하는 통계량이고, $(1-\alpha)100%$% $ 신뢰구간으로 추정할때 percentile interval은

$C^{PC}=[q_n^*(\alpha/2),q^*_n(1-\alpha/2)$ 이다.

이때 $q^*_n(\alpha)$는 부트스트랩 결과 나온 통계량 $\hat{\theta_{n1}^*}, \cdots{} ,\hat{\theta}_{nB}^*$ 그리고 최종적으로 percentile interval의 추정값은 $C^{PC}=[\hat{q_n}^*(\alpha/2),\hat{q_n}^*(1-\alpha/2)]$ 이 된다.

Percentile-t Interval

classic sampling t 통계량이 $T=\frac{\hat{\theta}-\theta}{s(\hat{\theta})}$일때, bootstrap의 t 통계량은 $T^*=\frac{\hat{\theta}^*-\hat{\theta}}{s^*(\hat{\theta})}$가 된다.

이때, 부트스트랩 통계량 $T^*$는 percentile interval 안에 들어와야 하므로 $q^*_{\alpha/2}\leq T^*\leq q^*_{1-\alpha/2}$로써 $q^*_{\alpha/2}\leq \frac{\hat{\theta}^*-\hat{\theta}}{s^*(\hat{\theta})}\leq q^*_{1-\alpha/2}$가 된다.

그러므로 percentile-t inverval은 $C^{pt}=[\hat{\theta}-s(\hat{\theta})q^*_{1-\alpha/2},\hat{\theta}-s(\hat{\theta})q^*_{\alpha/2}]$가 된다.

percentile-t interval은 다음 증명에 따라 asymptotically valid하다.

$$
\begin{aligned}
P[\theta \in C^{pt}] &=P[\hat{\theta}-s(\hat{\theta})q^*_{1-\alpha/2} \leq T^* \leq \hat{\theta}-s(\hat{\theta})q^*_{\alpha/2}]\\
&=P[q^*_{\alpha/2} \leq T^* \leq q^*_{1-\alpha/2}]\\
&=P[q_{\alpha/2} \leq T^* \leq q_{1-\alpha/2}]\\
&=1-\alpha
\end{aligned}
$$

---

### 6. Bootstrap Hypothesis Tests

Bootstrap 추정치의 p-value는 $p^*=\frac{1}{B}\Sigma_{b=1}^B1[|T^*(b)|>|T|]$가 된다.

하지만 bootstrap에서의 standard error는 구하기 힘드므로 non-studentized statistic $T=\hat{\theta}-\theta_0$를 이용하면 bootstrap의 non-studentized statistic은 $T^*=\hat{\theta}^*-\hat{\theta}$가 된다.

그리고 $100\alpha$%의 신뢰구간에서 $H_0: \theta=\theta_0$를 검증할때 bootstrap의 p-value는

$p^*=\frac{1}{B}\Sigma_{b=1}^B1[|\hat{\theta}^*(b)-\hat{\theta}|>|\hat{\theta}-\theta_0|]$가 된다.

---

### 7. Wald Statistic

만약 추정하고자 하는 모수 $\theta$가 하나가 아니라 vector라면 wald test를 해야한다.

$H_0:\theta=\theta_0,H_1:\theta\ne\theta_0$에 대해 bootstrap wald 통계량은

$W^*=(\hat{\theta}^*-\hat{\theta})'(\hat{V}_{\hat{\theta}})^{-1}(\hat{\theta}^*-\hat{\theta})$가 된다.

그리고 기각역 $W>q^*_{1-\alpha}$에 대해 p-value는 $p^*=\frac{1}{B}\Sigma_{b=1}^B1[W^*(b)>W]$이 된다.

하지만, $\hat{V}_{\hat{\theta}}$를 구하기 힘드므로 identity matrix를 사용하기도 한다.

Criterion Based Booststrap Tests

criterion-based estimator는 $\hat{\beta}=arg min_\beta J(\beta)$와 같은 제약식을 갖게 된다.

귀무가설 $H_0:\theta=\theta_0$에 대해 $\theta = r(\beta)$가 제약식이라면 $\tilde{\beta}=arg min_{r(\beta)=\theta_0}J(\theta)$가 되고

가설검증 criterion based statistic은 $J=min_{r(\beta)=\theta_0}J(\beta)-min_\beta J(\beta)=J(\tilde{\beta})-J(\hat{\beta})$이 되고 부트스트랩에서 제약식이 있는 경우 모수의 추정값은 

$\tilde{\beta}^*=arg min_{r(\beta)=\hat{\theta}} J^*(\beta)$이 된다. (단, $\hat{\theta}=r(\hat{\beta})$)

그리고 이때의 bootstrap J statistic은 $J^*=min_{r(\beta)=\hat{\theta}}J^*(\beta)-min_\beta J^*(\beta)=J^*(\tilde{\beta}^*)-J^*(\hat{\beta}^*)$이고, p-vlaue는 $p^*=\frac{1}{B}\Sigma_{b=1}^B1[J^*(b)>J]$가 된다.

---

### 8. Asymptotic Expansions

$T_n \in R$이 점진적으로 정규분포를 따를때 $T_n \rightarrow_dN(0,\sigma^2)$, $lim_{n\rightarrow \infty}G_n(u,F)=\Phi(\frac{u}{\sigma})$ 표준정규분포의 누적분포 함수가 되고, 0으로 수렴하는 $o(1)$을 따로 빼내면, $G_n(u,F)=\Phi(\frac{u}{\sigma})+o(1)$ 이와같이 된다.

즉, 통계량 $T_n$이 정규분포로 수렴하면 누적분포함수는 점진적으로 다음과 같다. 그리고 해당 함수를 Edgeworth Expansion이라고 한다.

$G_n(u,F)=\Phi(\frac{u}{\sigma})+\frac{1}{n^{1/2}}g_1(u,F)+\frac{1}{n}g_2(u,F)+O(n^{-3/2})$ 

해당 theorem을 적용하면,

1. $G_n(u,F)$을 $n^{1/2}$에 근사한다.
2. 그리고 second order of approximation에 의하여 $G_n(u,F) \simeq \Phi(\frac{u}{\sigma})+n^{-1/2}g_1(u,F)$가 된다.
3. 마지막으로 symmetric non-normal component를 추가해 $G_n(u,F) \simeq \Phi(\frac{u}{\sigma})+n^{(-1/2)}g_1(u,F)+n^{-1}g_2(u,F)$를 산출한다.

예를들어 $T_n=\sqrt{n}(\bar{X}_n-\mu)/\sigma$인 통계량이 있을때, 

$g_1(u)=-\frac{1}{6}k_3(u^2-1)\phi(u)$

$g_2(u)=-(\frac{1}{24}k_4(u^3-3u)+\frac{1}{72}k_3^2(u^5-10u^3+15u))\phi(u)$가 된다.

이때, $\phi(u)$는 standard normal pdf이고,

$k_3=E(X-\mu)^3/\sigma^3$,

$k_4=E(X-\mu)^4/\sigma^4-3$

---

### 9. One-Sided Tests

위의 표현식을 통해서 단특검정을 시행할 수 있다.

second order에 따라 $Pr(T_n < u) =G_n(u,F_0)=\Phi(u)+\frac{1}{n^{1/2}}g_1(u,F_0)+O(n^{-1})$는 exact distribution이 되고, 두 distribution의 차이는 

$$
\begin{aligned}
\Phi(u)-G_n(u,F_0) &=\frac{1}{n^{1/2}}g_1(u,F_0)+O(n^{-1})\\
&=O(n^{-1/2})
\end{aligned}
$$

이 된다.

그러므로 해당 모집단 분포의 오차는 $O(n^{-1/2})$ 가 된다.

그리고 bootstrap 분포는 

$G_n^*(u)=G_n(u,F_n)=\Phi(u)-\frac{1}{n^{1/2}}g_1(u,F_n)+O(n^{-1})$ 과 같고,

오차를 알아보기 위해 모집단의 분포를 차분하면

$G_n^*(u)-G_n(u,F_n)=\frac{1}{n^{1/2}}(g_1(u,F_n)-g_1(u,F_0))+O(n^{-1})$ 이 된다.

그리고 $g_1(u,F_n)-g_1(u,F_0) \approx \frac{\partial}{\partial F}g_1(u,F_0)(F_n-F_0)=O(n^{-1/2})$ 이 되므로 부트스트랩과 모집단의 오차 또한 $G_n^*(u)-G_n(u,F_0)=O(n^{-1})$ 또는 $Pr(T^*_n \leq u)=Pr(T_n \leq u) + O(n^{-1})$ 이 된다.

---

### 10. Bootstrap for Regression Models

선형회귀식 $y_i = x_i'\beta + e_i$ 에 대해 non-parametric bootstrap 샘플로 만들어진 선형회귀식 $y_i^* = x_i^{*'}\hat{\beta}+e^*_i$ 는 보통 $E(e_i^*\mid x_i^*) \ne 0$ 을 만족하지 못한다.

그렇기 때문에 해당 조건을 만족시키기 위해 $x_i^*$을 고정시키고 i.i.d.하게 $e^*_i$를 난수시킨다.

즉, $x_i^*=x_i$ 로 고정하고 오차 $e^*_i$ 는 OLS의 오차 $\left{ \hat{e}_1,\cdots{},\hat{e}_n \right}$ 으로부터 정규분포 $N(0,\hat{\sigma}^2)$ 로 resampling 된 것이다. 하지만 해당 작업이 매우 어렵기 때문에 아래와 같은 조건하에서 $e^*_i$ 를 추출하는게 일반적이다.

$E(e_i^*|x_i)=0$

$E(e_i^{*2}|x_i)=\hat{e}_i^2$

$E(e_i^{*3}|x_i)=\hat{e}_i^3$ 

그리고 보통 $e_i^*$ 는 두가지 확률분포로부터 추출되어진다.

$Pr(e_i^* = (\frac{1+\sqrt{5}}{2}\hat{e}_i)=\frac{\sqrt{5}-1}{2\sqrt{5}}$

$Pr(e_i^* = (\frac{1-\sqrt{5}}{2}\hat{e}_i)=\frac{\sqrt{5}+1}{2\sqrt{5}}$