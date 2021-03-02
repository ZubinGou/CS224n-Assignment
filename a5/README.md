## 1. Attention exploration (21 points)
multi-headed self-attention

### (a) (2 points) Copying in attention
当由于稀疏等原因使得 $q\perp k_j$，对所有j时

### (b) (4 points) An average of two
$$q = t(k_a+k_b)$$

$$t>0, \text{越大越近似}$$

### (c) (5 points) Drawbacks of single-headed attention:
#### i.
$$q = t(u_a+u_b)$$

#### ii.
由于$k_a$长度较长或者较短，使得$c$中$v_a$分量相对变长或变短，不稳定。

### (d) (3 points) Benefits of multi-headed attention
#### i.
$$q_1=\mu_a, q_2=\mu_b$$

#### ii.
由于引入了多头注意力，两个head计算后再平均，结果依然稳定满足：$c\approx\frac{1}{2}(v_a+v_b)$

### (e) (7 points) Key-Query-Value self-attention in neural networks
#### i.
$$c_2\approx u_a$$

不能。假设加上$u_d$，使得$\alpha_{21}$增大，即$x_1$权重增大，但$u_d$与$u_b$分量在$c_2$中会同时增大，所以无法近似$u_b$

#### ii.
$$V=u_bu_b^T\odot\frac{1}{\|u_b\|_{2}^{2}}-u_cu_c^T\odot\frac{1}{\|u_c\|_{2}^{2}}$$

$$K=I$$

$$Q=u_du_a^T\odot\frac{1}{\|u_a\|_{2}^{2}}-u_cu_d^T\odot\frac{1}{\|u_d\|_{2}^{2}}$$


## 2. Pretrained  Transformer models and knowledge access
直接训练Transformer来获取训练集之外的知识会失败，在Wikipedia text上预训练该Tranformer后则可以回答部分问题（本实验任务为回答单一问题：人物出生地）。

Based on: Andrej Karpathy’s [minGPT](https://github.com/karpathy/minGPT)



## 3. Considerations in pretrained knowledge (5 points)
