## 1. Machine Learning & Neural Networks
### (a) Adam Optimizer
#### i. momentum
动量的功能类似于滑动窗口平均，使得梯度m主要受到之前值的影响，就算当前梯度爆炸也会被稀释。
这种平滑方法减小了梯度变化程度，增加了模型稳定性，收敛更快；另外借助动量的惯性也可以逃出部分局部最优点。

#### ii. Adam
m是移动平均梯度（一阶动量），v是指数移动平均梯度（二阶动量）

Adam使得梯度趋于1，小梯度放大以逃离局部最优点，大梯度缩小以增加稳定性。
 

### (b) Dropout
#### i
$$
\gamma=\frac{1}{1-p_{\text {drop }}}
$$
证明：
$$
\sum_{i}\left[h_{\text {drop }}\right]_{i}=\gamma \sum_{i}\left(1-p_{\text {drop }}\right) h_{i}=\gamma\left(1-p_{\text {drop }}\right) E[h]=E[h]
$$

#### ii
评价模型的时候，dropout会产生随机性，禁用dropout可以展现模型性能和正则化（dropout）效果。

## 2. Neural Transition-Based Dependency Parsing

### (a)

| Stack                          | Buffer                                 | New dependency       | Transition           |   |
|--------------------------------|----------------------------------------|----------------------|----------------------|---|
| [ROOT]                         | [I, parsed, this, sentence, correctly] |                      | Initial Conﬁguration |   |
| [ROOT, I]                      | [parsed, this, sentence, correctly]    |                      | SHIFT                |   |
| [ROOT, I, parsed]              | [this, sentence, correctly]            |                      | SHIFT                |   |
| [ROOT, parsed]                 | [this, sentence, correctly]            | parsed → → I         | LEFT-ARC             |   |
| [ROOT, parsed, this]           | [sentence, correctly]                  |                      | SHIFT                |   |
| [ROOT, parsed, this, sentence] | [correctly]                            |                      | SHIFT                |   |
| [ROOT, parsed, sentence]       | [correctly]                            | sentence → → this    | LEFT-ARC             |   |
| [ROOT, parsed]                 | [correctly]                            | parsed → → sentence  | RIGHT-ARC            |   |
| [ROOT, parsed, correctly]      | []                                     |                      | SHIFT                |   |
| [ROOT, parsed]                 | []                                     | parsed → → correctly | RIGHT-ARC            |   |
| [ROOT]                         | []                                     | ROOT → → parsed      | RIGHT-ARC            |   |

### (b)
n shift + n arc = 2n

### (c)

### (d)

### (e)

### (f)
