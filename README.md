# 🏠 Deep Learning Buổi 2 — HOMES Dataset

## 📌 Giới thiệu

Bài thực hành xây dựng mô hình Deep Learning dự đoán giá nhà bằng TensorFlow/Keras kết hợp quy trình tiền xử lý dữ liệu chuẩn nhằm tránh **Data Leakage**.

Dataset sử dụng:
- `23_HOMES.csv`

Bài toán:
- Regression — dự đoán giá nhà dựa trên các thuộc tính của căn nhà.

---

# 🎯 Mục tiêu bài thực hành

- Hiểu và áp dụng đúng quy trình preprocessing trong Machine Learning
- Tránh lỗi Data Leakage khi scaling dữ liệu
- Sử dụng:
  - MinMaxScaler
  - OneHotEncoder
  - ColumnTransformer
- Xây dựng mô hình Neural Network Regression bằng TensorFlow/Keras
- Sử dụng:
  - Dropout
  - EarlyStopping
- Vẽ biểu đồ Loss Curve
- Đánh giá mô hình bằng:
  - MSE
  - MAE
  - R² Score

---

# 📂 Dataset

Dataset:
- `23_HOMES.csv`

Nguồn dữ liệu:

https://github.com/huynhhoc/DataAnalystDeepLearning/blob/main/Data/23_HOMES.csv

Dataset bao gồm các thuộc tính:
- diện tích
- số phòng
- tuổi nhà
- thuế
- số phòng ngủ
- số phòng tắm
- giá nhà

---

# 🛠️ Công nghệ sử dụng

- Python
- TensorFlow / Keras
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# ⚠️ Golden Rule chống Data Leakage

Quy tắc quan trọng trong preprocessing:

```text
Split FIRST
→ Fit ONLY on train
→ Transform train/test
→ Test ONLY once
```

Ý nghĩa:
- Không được fit scaler trên toàn bộ dataset
- Chỉ fit preprocessing trên train set
- Test set chỉ dùng để đánh giá cuối cùng

---

# 🔄 Flow quy trình thực hiện

```text
Raw Data (23_HOMES.csv)
        │
        ▼
1. Khám phá dữ liệu (EDA)
        │
        ▼
2. Split FIRST — train_test_split()
        │
        ├──── X_train ──────────────────────────────┐
        │         │                                  │
        │         ▼                                  │
        │   3. Fit Preprocessor                      │
        │      - MinMaxScaler (numeric)              │
        │      - OneHotEncoder (categorical)         │
        │         │                                  │
        │         ▼                                  ▼
        │   4. Transform Train          4. Transform Test
        │         │                                  │
        └─────────┴──────────────────────────────────┘
                  │
                  ▼
        5. Build Keras Model
           Input → Dense(32) → Dropout → Dense(16)
                 → Dropout → Dense(1)
                  │
                  ▼
        6. Train
           validation_split=0.2
           EarlyStopping
                  │
                  ▼
        7. Evaluate TEST SET
```

---

# 🧹 Tiền xử lý dữ liệu

## 1. Data Cleaning

Thực hiện:
- xóa khoảng trắng tên cột
- xóa dữ liệu null
- xóa dữ liệu trùng lặp

```python
df.columns = df.columns.str.strip()

df = df.dropna()

df = df.drop_duplicates()
```

---

# 📊 Chia dữ liệu Train/Test

```python
train_test_split(
    test_size=0.2,
    random_state=42
)
```

- 80% train
- 20% test

---

# 🔧 Preprocessing Pipeline

## Numeric Features

Sử dụng:

```python
MinMaxScaler()
```

Công thức:

\[
x' = \frac{x-x_{min}}{x_{max}-x_{min}}
\]

---

## Categorical Features

Sử dụng:

```python
OneHotEncoder()
```

Ví dụ:

| House_Type |
|---|
| Villa |
| Apartment |

Sau encoding:

| Villa | Apartment |
|---|---|
| 1 | 0 |
| 0 | 1 |

---

# 🏗️ Mô hình Deep Learning

Mô hình sử dụng:

```python
Sequential([
    Dense(32, activation='relu'),
    Dropout(0.2),

    Dense(16, activation='relu'),
    Dropout(0.2),

    Dense(1)
])
```

Kiến trúc:

```text
Input
 ↓
Dense(32)
 ↓
Dropout(0.2)
 ↓
Dense(16)
 ↓
Dropout(0.2)
 ↓
Dense(1)
```

---

# 🧠 Compile Model

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

## Optimizer
- Adam

## Loss Function

Mean Squared Error:

\[
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y_i})^2
\]

## Metric
- MAE

---

# ⏹️ EarlyStopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

Mục đích:
- tránh overfitting
- dừng train khi validation loss không cải thiện

---

# 📈 Loss Curve

Biểu đồ Loss Curve giúp theo dõi:
- quá trình học của model
- khả năng tổng quát hóa
- dấu hiệu overfitting

## Code vẽ Loss Curve

```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
```

## Kết quả

![Loss Curve](images/loss_curve.png)

---

# 📊 Đánh giá mô hình

Các metric sử dụng:

## MSE
- Mean Squared Error

## MAE
- Mean Absolute Error

## R² Score
- đánh giá độ phù hợp của mô hình

---

# 💾 Lưu Model & Preprocessor

## Save Model

```python
model.save("homes_regression_model.h5")
```

## Save Preprocessor

```python
joblib.dump(
    preprocessor,
    "homes_preprocessor.pkl"
)
```

---

# ✅ Kết quả đạt được

- Xây dựng thành công pipeline preprocessing chống Data Leakage
- Chuẩn hóa dữ liệu đúng quy trình
- Xây dựng mô hình Deep Learning Regression
- Huấn luyện mô hình bằng TensorFlow/Keras
- Sử dụng EarlyStopping chống overfitting
- Vẽ được Loss Curve
- Đánh giá mô hình bằng MSE, MAE và R² Score
- Lưu model và preprocessor thành công

---

# 📚 Kiến thức học được

- Quy trình preprocessing chuẩn trong Machine Learning
- Cách chống Data Leakage
- MinMax Scaling
- OneHot Encoding
- ColumnTransformer Pipeline
- Neural Network Regression
- Dropout
- EarlyStopping
- Loss Curve Analysis

---

# 📖 Tài liệu tham khảo

## TensorFlow
https://www.tensorflow.org/

## Scikit-learn
https://scikit-learn.org/

## Dataset
https://github.com/huynhhoc/DataAnalystDeepLearning

---

# 👨‍💻 Sinh viên thực hiện

Nguyễn Thị Minh Thư
