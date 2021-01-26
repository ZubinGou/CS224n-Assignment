## 1 Written: Understanding word2vec (23 points)

假设词典大小V，词向量长度D，
- 矩阵U和V为：D*V
- y和$\hat{y}$为：V*1

### (a) 
y为one-hot，只有$y_o$为1
$$
-\sum_{w \in \text { Vocab }} \boldsymbol{y}_{w} \log \left(\hat{y}_{w}\right)=-y_{o} \log \left(\hat{y}_{o}\right)-\sum_{w \in \text { Vocab }, w \neq o} y_{w} \log \left(\hat{y}_{w}\right)=-\log \left(\hat{y}_{o}\right)
$$
### (b)
$$
\begin{aligned}
\frac{\partial J\left(v_{c}, o, U\right)}{\partial v_{c}} &=-\frac{\partial\left(u_{o}^{T} v_{c}\right)}{\partial v_{c}}+\frac{\partial\left(\log \left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)\right)}{\partial v_{c}} \\
&=-u_{o}+\frac{1}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \frac{\partial\left(\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)\right)}{\partial v_{c}} \\
&=-u_{o}+\sum_{w} \frac{\exp \left(u_{w}^{T} v_{c}\right) u_{w}}{\sum_{w} \exp \left(u_{w}^{T} v_{c}\right)} \\
&=-u_{o}+\sum_{w} p(O=w \mid C=c) u_{w} \\
&=-u_{o}+\sum_{w} \hat{y}_{w} u_{w} \\
&=U(\hat{y}-y)
\end{aligned}
$$

### (c)
1. $w\neq 0$:
$$
\begin{aligned}
\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}} &=0+p(O=w \mid C=c) v_{c} \\
&=\hat{y}_{w} v_{c}
\end{aligned}
$$
2. $w=0$:
$$
\begin{aligned}
\frac{\partial J\left(v_{c}, o, U\right)}{\partial u_{w}} &=-v_{c}+p(O=o \mid C=c) v_{c} \\
&=\hat{y}_{w} v_{c}-v_{c} \\
&=\left(\hat{y}_{w}-1\right) v_{c}
\end{aligned}
$$
then:
$$
\frac{\partial J\left(v_{c}, o, U\right)}{\partial U}=v_{c}(\hat{y}-y)^{T}
$$

### (d)
$$
\begin{aligned}
\frac{\partial \sigma\left(x_{i}\right)}{\partial x_{i}} &=\frac{1}{\left(1+\exp \left(-x_{i}\right)\right)^{2}} \exp \left(-x_{i}\right)=\sigma\left(x_{i}\right)\left(1-\sigma\left(x_{i}\right)\right) \\
\frac{\partial \sigma(x)}{\partial x} &=\left[\frac{\partial \sigma\left(x_{j}\right)}{\partial x_{i}}\right]_{d \times d} \\
&=\left[\begin{array}{cccc}
\sigma^{\prime}\left(x_{1}\right) & 0 & \cdots & 0 \\
0 & \sigma^{\prime}\left(x_{2}\right) & \cdots & 0 \\
\vdots & \vdots & \vdots & \vdots \\
0 & 0 & \cdots & \sigma^{\prime}\left(x_{d}\right)
\end{array}\right] \\
&=\operatorname{diag}\left(\sigma^{\prime}(x)\right)
\end{aligned}
$$

### (e)
$$
\begin{aligned}
\frac{\partial J_{\text {negseample }}}{\partial v_{c}} &=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) u_{o}+\sum_{k=1}^{K}\left(1-\sigma\left(-u_{k}^{T} v_{c}\right)\right) u_{k} \\
&=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) u_{o}+\sum_{k=1}^{K} \sigma\left(u_{k}^{T} v_{c}\right) u_{k}
\end{aligned}
$$

$$
\frac{\partial J_{\text {neg-sample }}}{\partial u_{o}}=\left(\sigma\left(u_{o}^{T} v_{c}\right)-1\right) v_{c}
$$

$$
\frac{\partial J}{\partial u_{k}}=-\left(\sigma\left(-u_{k}^{\top} v_{c}\right)-1\right) v_{c}=\sigma\left(u_{k}^{\top} v_{c}\right) v_{c}, \quad \text { for } k=1,2, \ldots, K
$$

对比(b),(c)中softmax的偏导数可以看到，softmax反向传播时对输出矩阵(V * 1)以及词向量矩阵U进行了复杂的运算，而负采样复杂度与K有关，可以单独更新$v_c$, $u_o$和$u_k$而不必计算其他部分。

### (f)
$$
\frac{\partial J_{s g}}{\partial U} \quad=\sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial U}
$$

$$
\frac{\partial J_{s g}}{\partial v_{c}}=\sum_{-m \leq j \leq m, j \neq 0} \frac{\partial J\left(v_{c}, w_{t+j}, U\right)}{\partial v_{c}}
$$

$$
\frac{\partial J_{s g}}{\partial v_{w}}=0(\text { when } w \neq c)
$$