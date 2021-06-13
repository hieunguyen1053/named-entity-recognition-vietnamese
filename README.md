# Nhận dạng thực thể tên riêng tiếng Việt

**Người thực hiện**

- Nguyễn Trọng Hiếu ([@hieunguyen1053](https://github.com/hieunguyen1053))

## Kết quả thử nghiệm

Dữ liệu VLSP 2016: mức từ (không dùng nhãn gold POS, Chunk)

| Mô hình                                      | F1(%) |
| -------------------------------------------- | ----- |
| Hidden Markov Model (HMM)                    | 0.58  |
| Conditional Random Fields (CRF)              | 0.89  |
| Unidirectional Long short-term memory (LSTM) | 0.49  |
