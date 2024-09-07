n = int(input())  # Kích thước của ảnh (n x n)
a = []  # Mảng chứa ma trận ảnh
cnt, p = [0] * 256, [0] * 256  # Khởi tạo mảng đếm số lượng và xác suất của các mức xám

# Đọc vào ma trận ảnh
for i in range(n):
    b = list(map(int, input().split()))
    a.append(b)

# Đếm số lần xuất hiện của từng mức xám
for i in range(n):
    for j in range(n):
        cnt[a[i][j]] += 1

# Tính xác suất p[i] (xác suất xuất hiện của mỗi mức xám)
for i in range(256):
    p[i] = cnt[i] / (n * n)  # Xác suất của mức xám i

# Khởi tạo ngưỡng ban đầu t0 là trung bình trọng số của các mức xám
t0 = sum(i * p[i] for i in range(256))
t0 = int(t0)
epsilon = 0.001  # Sai số cho phép
while True:
    # Phân chia ảnh thành hai nhóm dựa trên ngưỡng hiện tại t0
    lower_sum, lower_weight = 0, 0
    upper_sum, upper_weight = 0, 0
    for i in range(256):
        if i <= t0:
            lower_sum += i * p[i]
            lower_weight += p[i]
        else:
            upper_sum += i * p[i]
            upper_weight += p[i]
    
    # Tính trung bình của mỗi nhóm
    if lower_weight > 0:
        mean_lower = lower_sum / lower_weight
    else:
        mean_lower = 0
    
    if upper_weight > 0:
        mean_upper = upper_sum / upper_weight
    else:
        mean_upper = 0
    
    # Cập nhật ngưỡng mới t1
    t1 = (mean_lower + mean_upper) / 2
    
    # Kiểm tra điều kiện dừng
    if abs(t1 - t0) < epsilon:
        break
    t0 = t1

# In kết quả ngưỡng cuối cùng (lấy phần nguyên)
print(f"Isodata threshold = {int(round(t1))}")


