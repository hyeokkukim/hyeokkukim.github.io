---
title:  "[FRM] Statistical and Probability Foundations"
excerpt: 
toc: true
toc_sticky: true

categories:
  - Financial Risk Management

use_math: true

---

# CH4_Statistical and probability foundations


## 1. Definition of Financial Price and Returns

- One-day absolute price change: $$D_t = P_t - P_{t-1}$$
- One-day relative return: $$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$
- One-day 로그수익률(or continuously-compounded return): $$r_t = ln(1+R_t) = ln(\frac{P_t}{P_{t-1}}) = ln(P_t) - ln(P_{t-1}) = p_t - p_{t-1}$$
- Multiple day 수익률(시차 = k): $$R_t(k) = \frac{P_t-P_{t-k}}{P_{t-k}}$$
- Multiple day gross return:  $$r_t(k) = ln[1+R_t(k)] = ln[(1+R_t)\cdot{}(1+R_{t-1})\cdot(1+R_{t-k-1}) = r_t + r_{t-1} + \cdots{} + r_{t-k-1}$$
- Percent and continuous compounding in aggregating returns
    
    어떤 포트폴리오의 최초 평가가격이 $$P_0$$ 일때, 해당 포트폴리오의 다음 period 후의 평가가격은
    
    - $$P_1 = w_1\cdot P_0\cdot e^{r_1}+w_2\cdot P_0 \cdot e^{r_2} + w_3\cdot P_0\cdot e^{r_3}$$
    - $$r_p = ln(w_1\cdot{}P_0\cdot{}e^{r_1}+w_2\cdot{}P_0\cdot{}e^{r_2} + w_3\cdot{}P_0\cdot{}e^{r_3})$$
    - $$P_1 = W_1\cdot{}P_0\cdot{}(1+r_1)+W_2\cdot{}P_0\cdot{}(1+r_2)+W_3\cdot{}P_0\cdot{}(1+r_3)$$
    - 포트폴리오 수익률: $$R_p = W_1\cdot{}r_1 + W_2\cdot{}r_2 + W_3\cdot{}r_3 = \frac{P_1 - P_0}{P_0}$$

## 2. Modeling Financial Prices and Returns

- Random walk for single-price
    - 어떤 자산의 $t$시점의 자산가격은 $$P_t = \mu + P_{t-1} + \sigma \epsilon_t$$, $$\epsilon_t$$ ~ $$N(0,1)$$
    - 로그수익률은 $$ln(P_t) = \mu + ln(P_{t-1}) + \sigma\epsilon_t$$
    - $$P_t = P_{t-1}exp(\mu+\sigma\epsilon_t)$$
- Random walk for fixed-income
    - 주식과는 다르게 채권은 가격(price)와 이자율(yield)가 있기 때문에, 이자율로 random walk를 계산
    - $$y_t = \mu + y_{t-1} + \sigma\epsilon_t$$
- Time dependent properties of the random walk model
    - Homoskedastic, 평균과 분산이 일정(constant): $$P_t = \mu + c\cdot{}P_{t-1}+\epsilon_t$$
    - Heteroskedastic, 평균과 분산이 변동(non-stationary): $$P_t = \mu_t + P_{t-1} + \epsilon_t$$
        - $$E_0[P_t|P_0] = P_t + \mu t$$
        - $$V_0[P_t|P_0] = \sigma^2 t$$
        

## 3. Investigating the Randomw Walk Model

- Autocorrelation
    - Standard Correlation of $$x,y$$: $$\rho_{xy} = \frac{\sigma^2_{xy}}{\sigma_x\sigma_y}$$, $$(\sigma^2_{xy} = E[(X-\mu_x)(Y-\mu_y)])$$
    - $k$th 수익률의 autocorrelation: $$\rho_k = \frac{\sigma^2_{t,t-k}}{\sigma_t\sigma_{t-k}} = \frac{\sigma^2_{t,t-k}}{\sigma^2_t}$$일때 sampling 수익률 $$r_t,t=1, \cdots{},T$$라면 autocorrelation의 추정값
        
        $$\hat{\rho}_k = \frac{\Sigma_{t=k+1}^T{(r_t-\bar{r})(r_{t-k}-\bar{r})/[t-(k-1)]}}{\Sigma_{t=1}^T(r_t-\bar{r})^2/(T-1)}$$
        
- Box-Ljung Statistic for daily log price changes: autocorrelation 검증을 위한 통계량
    - $$H_0:$$ time series is not autocorrelation
    - $$BL(P) = T\cdot{}(T+2)\Sigma_{k=1}^P\frac{\rho_k^2}{T-k}$$
    - 그리고, Box-Ljung 통계량은 카이제곱 분포를 따른다.
- Multivariate extensions
    - 두 자산 수익률 $$r_{1t}, r_{2t}$$에 대해 $$\sigma^2_{12t} = E\{[r_{1t}-E(r_{1t})][r_{2t}-E(r_{2t})]\} = E(r_{1t}r_{2t})- E(r_{1t})E(r_{2t}) $$

## 4. A review of historical observations of return distributions

- 수익률이 정규분포를 따를때
    - PDF: $$f(r_t) = \frac{1}{\sqrt{2\pi\sigma^2}}exp-(\frac{1}{2\sigma^2})(r_t-\mu)^2$$
    - Skewness: $$s^3 = E[(r_t-\mu)^3]$$
    - Skewness coefficient: $$r = \frac{E[(r_t-\mu)^3]}{\sigma^3}$$
    - Kurtosis(뾰족성): $$s^4 = E[(r_t-\mu)^4]$$
    - Kurtosis coefficient: $$k = E[(r_t-\mu)^4]/\sigma^4$$
- Percentiles for measuring market risk
    - standardized return: $$\tilde{r}_t = \frac{r_t-\mu_t}{\sigma_t}$$
    - $$Pr(\tilde{r}_t < -1.65)=Pr((r_t-\mu_t)/\sigma_t < -1.65) = Pr(r_t < -1.65\sigma_t+\mu_t)=5\%$$
- Aggregation in normal model
    
    포트폴리오 수익률 $$r_{pt} = w_1r_{1t} + w_2r_{2t} + w_3r_{3t}$$에 대해
    
    $$r_{1t} = \mu_1+\sigma_1 + \epsilon_{1t},r_{2t} = \mu_2+\sigma_2 + \epsilon_{2t},r_{3t} = \mu_3+\sigma_3 + \epsilon_{3t}$$라면
    
    $$\begin{bmatrix}
    \epsilon_{1t}\\
    \epsilon_{2t}\\\epsilon_{3t}\\\end{bmatrix} = MVN \begin{pmatrix}\begin{bmatrix} 0\\0\\0\\\end{bmatrix},\begin{bmatrix}1 & \rho_{12t}&\rho_{13t}\\ \rho_{21t}&1&\rho_{23t}\\ \rho_{31t}&\rho{32t}&1\end{bmatrix}\end{pmatrix} = (\mu_t,R_t)$$
    
    $$ r_t $$는 오차항의 correlation matrix, $$\mu_{pt} = w_1\mu_1 + w_2\mu_2 + w_3\mu_3,$$
    
    $$\sigma^2_{pt} = w_1^2 \sigma_{1t}^2 + w_2^2\sigma_{2t}^2 + w_3^2\sigma_{3t}^2 + 2w_1w_3\sigma^2_{13t} + 2w_2w_3\sigma^2_{23t}$$