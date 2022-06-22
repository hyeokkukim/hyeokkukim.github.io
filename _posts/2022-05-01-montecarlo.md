---
title:  "[Econ] Minimum Distance and Monte Carlo Simulation"
excerpt: "제약식이 있는 경우, 회귀계수 추정"
toc: true
toc_sticky: true

categories:
  - Econometrics

use_math: true

---

### 1. Criterion-Based Tests

Wald test는 비선형방정식에는 적절하지 않다. 그래서 이 경우 Criterion-Based Test $J(\beta)$를 사용한다.

Criterion-Based Test는 제약식이 있는 경우 $H_0: \beta \in B_0$와 제약식이 없는 경우 $H_1:\beta \notin B_0$ (단, $B_0 \in B$)로 나누어서 $J$ statistic을 구한다.

- 제약식이 없는 경우: $\hat{\beta}=argmin_{\beta \in B} J(\beta)$
- 제약식이 있는 경우: $\tilde{\beta}=argmin_{\beta \in B_0} J(\beta)$

그리고 추정된 $\beta$들로 구채해진 $J$statistic 차이인 criterion-based statistic $J$를 산출한다.

$$J = min_{\beta \in B_0}J(\beta)-min_{\beta \in B}J(\beta)

    =J(\tilde{\beta})-J(\hat{\beta})>0$$ 


이렇게 산출된 criterion-based statistic $J$는 minimum-distance statistic 또는 likelihood-ratio-like statistic이라고 한다.

---

### 2. Minimum Distance Estimator

Minimum distance estimator는 불편성을 만족하지 않는 파라미터를 최대한 불편성을 만족하도록 하는 것이다.

예를 들어 선형회귀모델 $y_i = x_i'\beta +e_i$의 coefficient $\hat{\beta}$가 불편성을 만족하지 않을때의 quadratic criterion function은 다음과 같다.

- $J(\beta) = n(\hat{\beta}-\beta)'\hat{W}(\hat{\beta}-\beta)$
- (단, $\hat{W}$는 $k\times k$의 positive weight matrix)

결과의 해석은 $J(\beta)$가 작은 경우 추정값 $\hat{\beta}$은 모수 $\beta$에 가깝다는 것이다. 

그리고 해당 function을 minimize 해서 우리는 모수 추정값 $\tilde{\beta}_{md}$ 을 찾을 수 있다.

$$\tilde{\beta}_{md}=argmin_{R'\beta=r}J(\beta)$$

$\tilde{\beta}_{md}$를 찾는 방법은 Lagrangian 방법으로 찾을 수 있다. Largrange multipliers는 $\mathcal{L}(\beta,\lambda)=\frac{1}{2}J(\beta,\hat{W})+\lambda'(R'\beta-r)$에서 $(\beta,\lambda)$를 최소화 하는 것이다.

그러므로 Largrangian 방법을 사용해서 $\tilde{\beta}_{md}$를 찾는것이다.

- $\tilde{\lambda}_{md}=n(R'\hat{W}^{-1}R)^{-1}(R'\hat{\beta}-r)$
- $\tilde{\beta}_{md}=\hat{\beta}-\hat{W}^{-1}R(R'\hat{W}^{-1}R)^{-1}(R'\hat{\beta}-r)$
- assumption $R'\beta=r(R\text{ is } k\times q,rank(R)=q)$
- assumption $\hat{W}\rightarrow_pW>0$

$\therefore \tilde{\beta}_{md}\rightarrow_p\beta$로 불편성을 만족하게 된다.

또한, 이는 $$\sqrt{n}(\tilde{\beta}_{md}-\beta)\rightarrow_d N(0,V_\beta(W))$$ 
asymptotic normality를 만족하게 된다.

(단, $V_\beta(W)=V_\beta-W^{-1}R(R'W^{-1}R)^{-1}R'V_\beta-V_\beta R(R'W^{-1}R)^{-1}R'W^{-1}$\\
$+W^{-1}R(R'W^{-1}R'V_\beta R(R'W^{-1}R)^{-1}R'W^{-1},$\\
$V_\beta = Q^{-1}\Omega Q^{-1}=n\cdot{}(X'X)^{-1}(X'ee'X)(X'X)^{-1}$)

Efficient한 asymptotic optimal weight matrix는 asymptotic variance $V_\beta(W)$를 최소화 하는 것이다. 즉, $W=V_\beta^{-1}$일때 minimum distance estimator $\hat{W}=\hat{V}_\beta^{-1}$이 된다. 
그 결과 efficient minimum distance estimator는

$$\tilde{\beta}_{emd} = \hat{\beta}-\hat{V}_{\beta}R(R'\hat{V}_\beta R)^{-1}(R'\hat{\beta}-r)$$이 되고, asymptotic normality를 만족한다.

$$\sqrt{n}(\tilde{\beta}_{emd}-\beta)\rightarrow_dN(0,V_{\beta,emd})\text{ where } V_{\beta,emd} =V_\beta-V_\beta R(R'V_\beta R)^{-1}R'V_\beta$$

위에서 구한 식들을 통해 다음과 같은 relationship을 구할 수 있고, 

$$\sqrt{n}(\hat{\beta}-\tilde{\beta}_{emd})=\hat{V}_\beta R(R'\hat{V}_\beta R)^{-1}\sqrt{n}(R'\hat{\beta}-r)\rightarrow_dN(0,V_\beta R(R'V\beta R)^{-1} R'V_\beta)$$

asymptotic variance로 relationship을 재구성 하면 이는 efficient와 inefficient estimator의 차이인 Hausman Equality라고 부른다.

$$avar(\hat{\beta}-\tilde{\beta}_{emd})=avar(\hat{\beta})-avar(\tilde{\beta}_{emd})$$

---

### 3. Minimum Distance Tests

$H_0:R'\beta=\theta_0$일때 efficient minimum distance estimator $\tilde{\beta}_{emd}$가 제약조건 $R'\beta=\theta_0$일때 다음 추정값을 만족한다.

$$\tilde{\beta}_{emd}=\hat{\beta}-\hat{V}_\beta R(R'\hat{V}_{\beta}R)^{-1}(R'\hat{\beta}-\theta_0)$$\\
$$\hat{\beta}-\tilde{\beta}_{emd}=\hat{V}_\beta R(R'\hat{V}_\beta R)^{-1}(R'\hat{\beta}-\theta_0)$$

그리고, efficient minimum statistic은 Wald 통계량과 동일하다. 즉, 선형제약조건이 있을때 가장 efficient한 minimum statistic은 <span class="evidence">wald statistic</span> 이다.

$$
\begin{aligned}
J^* &= n(\hat{\beta}-\tilde{\beta}_{emd})'\hat{V}_\beta^{-1}(\hat{\beta}-\tilde{\beta}_{emd})\\

&=n(R'\hat{\beta}-\theta_0)'(R'\hat{V}_\beta R)^{-1} R'\hat{V}_\beta\hat{V}_\beta^{-1}\hat{V}_\beta R (R'\hat{V}_\beta R)^{-1}(R'\hat{\beta} -\theta_0)\\

&=n(R'\hat{\beta}-\theta_0)'(R'\hat{V}_\beta R)^{-1}(R'\hat{\beta}-\theta_0)\\

&=W
\end{aligned}
$$

그리고, assumption $h(\beta):R^k\rightarrow R^q,V_\theta = R'V_\beta R>0$ 조건하에

$J^*\rightarrow_dx^2_q$가 된다.
---
### 4. Minimum Dsitance Tests Under Homoskedasticity

만약 $\hat{W}=(\hat{V}_\beta^0)^{-1}=\hat{Q}/s^2=(\frac{1}{n}X'X)/s^2$ 일때 $J^0(\beta)=n(\hat{\beta}-\beta)'\hat{\Omega}(\hat{\beta}-\beta)/s^2$가 된다.

그리고, $H_0:\beta \in B_0$에서의 minimum distance statistic은 $J^0 = min_{\beta \in B_0}J^0(\beta)$가 된다.

만약 $\beta$의 SSE가 아래와 같다면
$$
\begin{aligned}
SSE(\beta)&=\Sigma_{i=1}^n(y_i-x'_i\beta)^2=\Sigma_{i=1}^n(x_i'\hat{\beta}+\hat{e}_i-x_i'\beta)^2\\
&=\Sigma_{i=1}^n\hat{e}_i^2+(\hat{\beta}-\beta)'(\Sigma_{i=1}^nx_ix_i')(\hat{\beta}-\beta)\\
&=n\hat{\sigma}^2+J^0(\beta)s^2
\end{aligned}
$$

$$\tilde{\beta}_{cls}=argmin_{\beta\in B_0}J^0{\beta}=argmin_{\beta \in B_0}SSE(\beta)$$
를 만족하는 estimator를 constrained lest-squares estimator라고 한다.

그리고, 검정통계량 J는 다음과 같다.
$$J^0=J^0(\tilde{\beta}_{cls})=n(\hat{\beta}-\tilde{\beta}_{cls})'\hat{Q}(\hat{\beta}-\tilde{\beta}_{cls})/s^2$$
---
### 5. F Tests

귀무가설 $H_0: \beta \in B_0$에서 F 통계량은 $F=\frac{(\tilde{\sigma}^2-\hat{\sigma}^2)/1}{\hat{\sigma}^2/(n-k)}$이고, 이때 $\hat{\sigma}^2=\frac{1}{n}\Sigma_{i=1}^n(y_i-x_i'\hat{\beta})^2.\tilde{\sigma}^2=\frac{1}{n}\Sigma_{i=1}^n(y_i-x_i'\tilde{\beta}_{cls})^2$이 된다.

F 통계량을 앞서 배운 내용들로 간소화 하면, 

$$F=\frac{SSE(\tilde{\beta}_{cls})-SSE(\hat{\beta})}{qs^2}=\frac{J^0}{q}(\text{where } SSE(\beta)=\Sigma_{i=1}^n(y_i-x_i'\beta)^2)$$

(단, $SSE(\beta)=\Sigma_{i=1}^n(y_i-x_i'\beta)^2$ ) 

가 되는데 이는 minimum distance statistic을 제약식의 개수 q로 나눈것과 같아진다.
---
### 6. Hausman Tests

Hausma test 는 두개의 추정량을 비교할때 사용한다.

Unconstrained least-squre estimator인 $\hat{\beta}$과 efficient minimum distance estimator인 $\tilde{\beta}_{emd}$에 대하여 아래의 가설을 검증한다고 하자.

$H_0: h(\beta)=\theta_0, H_1:h(\beta)\ne \theta_0$ 

두 추정값이 귀무가설하에 불편성을 만족하고, 대립가설하에 $\hat{\beta}$만 만족한다 할때 두 추정량 차이의 asymptotic distribution은 다음과 같다.

$$\sqrt{n} (\hat{\beta}-\tilde{\beta}_{emd})\rightarrow_dN(0,V_\beta R(R'V_\beta R)^{-1}R'V_\beta$$

그리고 이에 따른 Hausman 통계량은

$$
\begin{aligned}
H &= (\hat{\beta}-\tilde{\beta}_{emd})'avar(\hat{\hat{\beta}-\tilde{\beta}_{emd}})^{-}(\hat{\beta}-\tilde{\beta}_{emd})\\

&=n(\hat{\beta}-\tilde{\beta}_{emd})'(\hat{V}_\beta \hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'\hat{V}_\beta)^{-}(\hat{\beta}-\tilde{\beta}_{emd})\\
\end{aligned}
$$

이때, 행렬 $\hat{V}_\beta^{1/2}\hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'\hat{V}^{1/2}_\beta$ 는 idempotent 하므로 이 matrix의 generalized inverse(and Moore-Penrose inverse)는 자기 자신이 된다.

$$
\begin{aligned}
(\hat{V}_\beta\hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'\hat{V}_\beta)^{-} &=\hat{V}_\beta^{-1/2}(\hat{V}_\beta^{1/2}\hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'\hat{V}^{1/2}_\beta)^{-}\hat{V}_\beta^{-1/2}\\

                                                &=\hat{V}_\beta^{-1/2}\hat{V}_\beta^{1/2}\hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'\hat{V}^{1/2}_\beta\hat{V}_\beta^{-1/2}\\

                                                &=\hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'
\end{aligned}
$$

그러므로 Hausman statistic은 $H=n(\hat{\beta}-\tilde{\beta}_{emd})'\hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'(\hat{\beta}-\tilde{\beta}_{emd})\rightarrow_d\mathcal{X}^2_q$ 가 된다.

만약, 제약조건이 $\hat{R}=R,R'\tilde{\beta}_{emd}=\theta_0$ 를 만족하는 선형 제약조건이라면

$$
\begin{aligned}

H &=n(\hat{\beta}-\tilde{\beta}_{emd})'\hat{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}\hat{R}'(\hat{\beta}-\tilde{\beta}_{emd})\\

&=n(\hat{\beta}-\tilde{\beta}_{emd})'{R}(\hat{R}'\hat{V}_\beta\hat{R})^{-1}{R}'(\hat{\beta}-\tilde{\beta}_{emd})\\

&=n(R'\hat{\beta}-R'\tilde{\beta}_{emd})'(\hat{R}'\hat{V}_\beta\hat{R})^{-1}(R'\hat{\beta}-R'\tilde{\beta}_{emd})\\

&=n(R'\hat{\beta}-\theta_0)'(\hat{R}'\hat{V}_\beta\hat{R})^{-1}(R'\hat{\beta}-\theta_0)

\end{aligned}
$$

Hausman statistic은 Wald statistic과 동일하게 된다.

---

### 7. Monte Carlo Simulation

모집단 F의 관측치 $(y_i;x_i')$, 모수 $\theta$에 대해 estimator $\hat{\theta}$, t-statistic $(\hat{\theta}-\theta)/se(\hat{\theta})$ 와 같은 통계량의 집합을  $T_n=T_n{(y_i,x_1'),\cdots{},(y_n,x'_n),\theta}$ 이라고 하자.

이때, $T_n$ 의 누적분포 함수는 $G_n(u,F)=P(T_n\leq u$|$F)$ 라고 할때, $T_n$ 의 asymptotic distribution은 보통 알지만, $G_n$ 의 분포는 보통 알지 못한다.

그러므로 monte carlo simulation을 통해 특정 $F$를 선택하여 $G_n$의 분포를 추정하는 것이다.

몬테카를로 시뮬레이션은 다소 심플하다.

1. 연구자가 관측치에 대한 특정 $F$ 분포와 추정할 모수를 선택한다.
2. 그리고, n개의 random pairs $(y_i^*,x_i^{*'})$ 를 분포 $F$로부터 random하게 generating한다.
3. 추정하고자 하는 통계량 $T_n$ 을 계산한다.
4. 2~3번을 $B$ 번 만큼 반복해 총 $B$ 개의 통계량 $T_{nb},b=1,\cdots,B$ 를 산출한다.
5. random sample of size $B$ 의 distribution $G_n(u,F)$ 에 대해 $\hat{P}=\frac{1}{B}\Sigma_{b=1}^{B}1(T_{nb}\geq u)$ 값을 계산한다.