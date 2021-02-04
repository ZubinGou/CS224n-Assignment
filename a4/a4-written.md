## 1. Neural Machine Translation with RNNs (45 points)
Bidirectional LSTM Encoder + Unidirectional LSTM Decoder
Spanish to English

#### (g) enc_masks
enc_masks (b, src_len) 用于标记batch中每个句子中<pad>的位置为1

(1)作用：将注意力分数e_t中对应<pad>填充的部分设为$-\inf$，经过softmax后概率近乎为0

(2)为什么：可以屏蔽非句子本身的填充部分（填充是为了构造batch），将注意力集中到句子上

## 2. Analyzing NMT Systems (30 points)