# 🎙️ Speech Emotion Recognition with Spiking Neural Network (SER-SNN)

## 📌 Giới thiệu (Overview)
Dự án này đề xuất một mô hình lai giữa **Mạng nơ-ron tích chập (CNN)** và **Mạng nơ-ron xung (SNN)** để giải quyết bài toán Nhận dạng Cảm xúc qua Giọng nói (Speech Emotion Recognition - SER). 

Mô hình tận dụng sức mạnh của CNN trong việc trích xuất đặc trưng và SNN (sử dụng nơ-ron Leaky Integrate-and-Fire) cho bước phân loại cuối cùng. Giải pháp này không chỉ duy trì độ chính xác cao mà còn giảm thiểu số lượng phép toán dấu phẩy động (FLOPs), mang lại hiệu quả năng lượng vượt trội và phù hợp để triển khai trên các phần cứng Neuromorphic.

## 👥 Nhóm phát triển (Team)
Dự án thuộc lớp **CS338_P22**, **Nhóm 7** - Trường Đại học Công nghệ Thông tin (UIT):
* **Lê Hoàng Việt**
* **Hoàng Công Chiến**

## 📊 Tập dữ liệu (Datasets)
Mô hình được huấn luyện và đánh giá trên 4 tập dữ liệu cảm xúc giọng nói chuẩn:
* **Nguồn**: RAVDESS, TESS, CREMA-D, và SAVEE.
* **Nhãn cảm xúc**: Dữ liệu được đồng nhất về 5 nhãn chung: **Neutral, Happy, Sad, Angry, Fearful**.
* **Số lượng**: Tổng cộng 10.580 mẫu âm thanh (mỗi mẫu được chuẩn hóa độ dài 3 giây, 16kHz) đã qua làm sạch và cân bằng.

## 🧠 Kiến trúc mô hình (Architecture)
Kiến trúc lai CNN-SNN trải qua các giai đoạn xử lý sau:

1. **Trích xuất đặc trưng (Feature Extraction)**: Trích xuất song song và kết hợp (Stacked) 13 hệ số MFCC, 1 hệ số năng lượng RMSE và 1 hệ số ZCR theo từng khung thời gian. Kích thước tensor đầu vào: `15 x 299`.
2. **CNN Front-end**: Gồm 3 khối Conv2D (kernel `3x3`), kết hợp Batch Normalization, ReLU và Max Pooling (`2x2`) để trích xuất đặc trưng không gian - thời gian, sau đó làm phẳng thành vector 128 chiều.
3. **Mã hóa xung (Spike Encoding)**: Chuyển đổi vector đặc trưng thành chuỗi xung nhị phân bằng phương pháp **Rate Coding**.
4. **SNN Back-end (Phân loại)**: 3 lớp Fully Connected sử dụng mô hình nơ-ron Leaky Integrate-and-Fire (LIF), với số lượng nơ-ron giảm dần: `128 -> 64 -> 32 -> 5` (tương ứng 5 lớp cảm xúc).

## 📈 Kết quả nổi bật (Results)
* **Đặc trưng tối ưu**: Việc kết hợp `MFCC + RMSE + ZCR` mang lại hiệu suất vượt trội (**71.9%**) so với việc chỉ dùng phương pháp MFCC thông thường.
* **Độ chính xác**: Sau khi tinh chỉnh bằng Surrogate Gradient, mô hình lai SNN đạt **độ chính xác 87.8%**, chỉ kém mô hình CNN gốc 0.5% nhưng tiết kiệm năng lượng tính toán hơn rất nhiều.

## 🛠 Cài đặt & Môi trường (Installation)
Dự án được xây dựng và huấn luyện trên GPU NVIDIA RTX 3090. Các thư viện cốt lõi bao gồm PyTorch và snnTorch.

```bash
# Clone the repository
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name

# Cài đặt các thư viện yêu cầu
pip install torch torchvision torchaudio
pip install snntorch librosa numpy pandas scikit-learn
