# LAB 2 - LINEAR REGRESSION WITH TENSORFLOW

## Thông tin sinh viên

- Họ tên: Nguyễn Thị Minh Thư
- Môn học: Thực hành TensorFlow / PyTorch
- Nội dung: Linear Regression với TensorFlow
- Dataset: 23_HOMES.csv

---

# 1. Mục tiêu bài thực hành

Trong buổi thực hành này tiến hành:

- Đọc dữ liệu từ file CSV
- Làm sạch dữ liệu
- Chuẩn hóa dữ liệu bằng Z-score
- Xây dựng mô hình Linear Regression bằng TensorFlow
- Huấn luyện mô hình
- Đánh giá mô hình
- Vẽ Loss Curve
- Sử dụng EarlyStopping
- Lưu scaler và model

---

# 2. Dataset sử dụng

Dataset được lấy từ GitHub:

https://github.com/huynhhoc/DataAnalystDeepLearning/blob/main/Data/23_HOMES.csv

Dataset chứa các thuộc tính của nhà ở như:
- diện tích
- số phòng
- vị trí
- năm xây dựng
- giá nhà

Mục tiêu:
- sử dụng các thuộc tính để dự đoán giá nhà.

---

# 3. Công nghệ sử dụng

- Python
- TensorFlow
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

---

# 4. Quy trình thực hiện

## Bước 1: Đọc dữ liệu

Sử dụng pandas để đọc file CSV:

```python
df = pd.read_csv(url)
```

---

## Bước 2: Làm sạch dữ liệu

Các bước thực hiện:
- xóa khoảng trắng tên cột
- kiểm tra dữ liệu null
- xóa dữ liệu null
- xóa dữ liệu trùng lặp

```python
df = df.dropna()
df = df.drop_duplicates()
```

---

## Bước 3: Chia dữ liệu Input và Output

```python
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
```

- `X`: dữ liệu đầu vào
- `y`: giá nhà cần dự đoán

---

## Bước 4: Chia train/test

```python
train_test_split(test_size=0.2)
```

- 80% dữ liệu train
- 20% dữ liệu test

---

## Bước 5: Chuẩn hóa dữ liệu

Sử dụng phương pháp Z-score:

\[
z = \frac{x - \mu}{\sigma}
\]

```python
scaler = StandardScaler()
```

Tiến hành:
- fit_transform() cho train set
- transform() cho test set

---

## Bước 6: Xây dựng mô hình Linear Regression

Mô hình được xây dựng bằng TensorFlow:

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
```

Công thức Linear Regression:

\[
\hat{y}=w_1x_1+w_2x_2+...+w_nx_n+b
\]

Trong đó:
- \(x\): input
- \(w\): trọng số
- \(b\): bias
- \(\hat{y}\): giá trị dự đoán

---

## Bước 7: Compile model

```python
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)
```

- `adam`: thuật toán tối ưu
- `mse`: hàm mất mát
- `mae`: metric đánh giá

---

## Bước 8: EarlyStopping

```python
EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

Ý nghĩa:
- theo dõi validation loss
- dừng train khi model không còn cải thiện
- tránh overfitting

---

## Bước 9: Huấn luyện mô hình

```python
model.fit(...)
```

Thông số:
- epochs = 100
- batch_size = 8

---

## Bước 10: Đánh giá mô hình

Các metric sử dụng:
- MSE
- MAE
- R² Score

```python
mean_squared_error()
mean_absolute_error()
r2_score()
```

---

# 5. Loss Curve

Biểu đồ Loss Curve dùng để theo dõi quá trình học của mô hình.

- Train Loss giảm:
  mô hình học tốt.

- Validation Loss tăng:
  dấu hiệu overfitting.

## Code vẽ Loss Curve

```python
plt.figure(figsize=(10,5))

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()
```

## Kết quả Loss Curve

![Loss Curve](images/loss_curve.png)

---

# 6. Kết quả đạt được

- Đọc và xử lý thành công dataset
- Chuẩn hóa dữ liệu bằng Z-score
- Xây dựng mô hình Linear Regression bằng TensorFlow
- Huấn luyện mô hình dự đoán giá nhà
- Vẽ được biểu đồ Loss Curve
- Sử dụng EarlyStopping để giảm overfitting
- Lưu scaler và model thành công

---

# 7. Kết luận

Qua bài thực hành đã hiểu được quy trình cơ bản của Machine Learning:

- Tiền xử lý dữ liệu
- Chuẩn hóa dữ liệu
- Xây dựng mô hình
- Huấn luyện mô hình
- Đánh giá mô hình

Ngoài ra cũng hiểu được vai trò của:
- StandardScaler
- Loss Function
- EarlyStopping
- Loss Curve

TensorFlow hỗ trợ xây dựng và huấn luyện mô hình hiệu quả, kết hợp với pandas và scikit-learn giúp xử lý dữ liệu thuận tiện hơn.

---

# 8. Tài liệu tham khảo

- TensorFlow Documentation:
  https://www.tensorflow.org/

- Scikit-learn Documentation:
  https://scikit-learn.org/

- Dataset:
  https://github.com/huynhhoc/DataAnalystDeepLearning
