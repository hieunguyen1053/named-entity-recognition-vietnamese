# Nhận dạng thực thể tên riêng tiếng Việt

**Người thực hiện**

- Nguyễn Trọng Hiếu ([@hieunguyen1053](https://github.com/hieunguyen1053))

## Dataset

|              | Train set | Dev set | Test set |
| ------------ | :-------: | :-----: | :------: |
| Sentences    |   14861   |  2000   |   2831   |
| Unique words |   18123   |  5735   |   7733   |

Num tags : 9

List tags : O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG, B-MISC, I-MISC

## F measure

F1 = 2 \* P \* R/(P + R)

với P (Precision) và R (Recall) được tính bằng công thức sau:

- P = NE-true/NE-sys
- R = NE-true/NE-ref

với:

- NE-ref: The number of NEs in gold data
- NE-sys: The number of NEs in recognizing system
- NE-true: The number of NEs which is correctly recognized by the system

## Kết quả thử nghiệm

Dữ liệu VLSP 2016: mức từ (không dùng nhãn gold POS, Chunk)

| Mô hình                                                                        | F1(%) |
| ------------------------------------------------------------------------------ | ----- |
| Hidden Markov Model (HMM)                                                      | 58.23 |
| Conditional Random Fields (CRF)                                                | 89.51 |
| Bidirectional Long short-term memory (LSTM)                                    | 72.17 |
| Bidirectional Long short-term memory - Conditional Random Fields (Bi-LSTM-CRF) | 76.72 |

## Tham khảo

1. [VLSP 2016 - Named Entity Recognition](https://vlsp.org.vn/vlsp2016/eval/ner)
2. [Named Entity Recognition (NER) using BiLSTM CRF](https://github.com/Gxzzz/BiLSTM-CRF)
3. [Nhận dạng thực thể tên riêng tiếng Việt](https://github.com/undertheseanlp/ner)
