import tkinter as tk
from tkinter import filedialog, Label, Button, ttk
from PIL import Image, ImageTk
import numpy as np
import math 

class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ứng dụng áp dụng các thuật toán Xử lý ảnh")
        self.root.geometry("1000x700")
        self.root.configure(bg="#F0F0F0")
        self.img = None
        self.img_array = None
        self.setup_gui()

    def setup_gui(self):
        # Tiêu đề
        title_label = Label(self.root, text="TRỰC QUAN CÁC THUẬT TOÁN XỬ LÝ ẢNH", 
                            font=("Arial", 24, "bold"), bg="#F0F0F0", fg="#333333")
        title_label.grid(row=0, column=0, columnspan=2, pady=20, sticky="nsew")
        self.root.columnconfigure(0, weight=1)

        # Frame chứa ảnh gốc và ảnh đã xử lý
        image_frame = tk.Frame(self.root, bg="#F0F0F0")
        image_frame.grid(row=1, column=0, columnspan=2, padx=20, pady=10, sticky="nsew")
        
        # Cấu hình các cột của image_frame để căn giữa
        image_frame.columnconfigure(0, weight=1)
        image_frame.columnconfigure(1, weight=1)

        # Label hiển thị ảnh gốc
        self.label_original = Label(image_frame, text="Ảnh gốc", 
                                    font=("Arial", 14), bg="#D3D3D3", relief="solid", bd=2)
        self.label_original.grid(row=0, column=0, padx=20, pady=10, sticky="nsew")

        # Label hiển thị ảnh sau khi xử lý
        self.label_filtered = Label(image_frame, text="Ảnh sau khi xử lý", 
                                    font=("Arial", 14), bg="#D3D3D3", relief="solid", bd=2)
        self.label_filtered.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

        # Frame cho các controls
        control_frame = tk.Frame(self.root, bg="#F0F0F0")
        control_frame.grid(row=2, column=0, columnspan=2, pady=20)

        # Nút tải ảnh
        btn_load = Button(control_frame, text="Tải ảnh lên", 
                        command=self.load_image, font=("Arial", 12),
                        bg="#4CAF50", fg="white", width=15)
        btn_load.pack(side=tk.LEFT, padx=10)

        # Dropdown cho lựa chọn bộ lọc
        self.filter_var = tk.StringVar()
        self.filter_choices = {
            "01. Biến đổi âm bản": self.negative,
            "02. Phân ngưỡng": self.apply_threshold,
            "03. Biến đổi tuyến tính": self.linear_transformation, 
            "04. Cân bằng lược đồ xám": self.histogram_equalization,
            "05. Lọc trung bình 3x3": self.apply_mean_filter,
            "06. Lọc trung vị 3x3": self.apply_median_filter,
            "07. Khuếch tán lỗi 1 chiều": self.apply_error_diffusion,
            "08. Phát hiện biên (Sobel)": self.apply_sobel,
            "09. Phát hiện biên (Prewitt)": self.apply_prewitt,
            "10. Phát hiện biên (Laplace)": self.apply_laplace,
            "11. Co ảnh": self.apply_erosion,
            "12. Dãn ảnh": self.apply_dilation,
            "13. Phân ngưỡng Otsu": self.apply_otsu
        }
        
        self.filter_dropdown = ttk.Combobox(control_frame, 
                                            textvariable=self.filter_var,
                                            values=list(self.filter_choices.keys()),
                                            font=("Arial", 12),
                                            width=25)
        self.filter_dropdown.set("Chọn phép toán xử lý")
        self.filter_dropdown.pack(side=tk.LEFT, padx=10)

        # Nút áp dụng bộ lọc
        btn_apply = Button(control_frame, text="Áp dụng", 
                        command=self.apply_selected_filter,
                        font=("Arial", 12), bg="#008CBA", fg="white", width=15)
        btn_apply.pack(side=tk.LEFT, padx=10)

        # Frame cho thanh trượt ngưỡng (ẩn khi mở app)
        self.threshold_frame = tk.Frame(self.root, bg="#F0F0F0")
        self.threshold_frame.grid(row=3, column=0, columnspan=2, pady=10)
        self.threshold_frame.grid_forget()  # Ẩn thanh trượt ban đầu

        # Thanh trượt ngưỡng
        self.threshold_slider = tk.Scale(self.threshold_frame, from_=0, to=255, orient="horizontal", 
                                          label="Ngưỡng", font=("Arial", 12), length=400, showvalue=128)
        self.threshold_slider.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = Image.open(file_path)
            self.display_resized_image(self.img)

    def apply_selected_filter(self):
        if self.img is not None and self.filter_var.get() in self.filter_choices:
            filter_func = self.filter_choices[self.filter_var.get()]
            filter_func()

    def display_resized_image(self, img):
        """
        Hàm này sẽ điều chỉnh kích thước ảnh sao cho vừa với khung hình hiển thị.
        """
        # Kích thước khung hiển thị (400x400 px)
        max_size = (400, 400)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Chuyển đổi ảnh đã thay đổi kích thước sang PhotoImage
        img_tk = ImageTk.PhotoImage(img)

        # Hiển thị ảnh gốc
        self.label_original.config(image=img_tk)
        self.label_original.image = img_tk
        self.label_original.photo = img_tk

    def display_result(self, result_array):
        """
        Hiển thị kết quả xử lý ảnh
        Params:
            result_array: Mảng numpy chứa ảnh đã xử lý
        """
        # Chuyển ma trận mức xám thành ảnh PIL
        filtered_img = self.gray_matrix_to_image(result_array)
        
        # Resize ảnh để hiển thị phù hợp
        self.display_resized_image_for_filtered(filtered_img)

    def display_resized_image_for_filtered(self, img):
        """
        Hàm này sẽ điều chỉnh kích thước ảnh sao cho vừa với khung hình hiển thị và hiển thị ảnh đã xử lý
        ở cột 2.
        """
        # Kích thước khung hiển thị (400x400 px)
        max_size = (400, 400)
        img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Chuyển đổi ảnh đã thay đổi kích thước sang PhotoImage
        img_tk = ImageTk.PhotoImage(img)

        # Hiển thị ảnh sau khi xử lý
        self.label_filtered.config(image=img_tk)
        self.label_filtered.image = img_tk
        self.label_filtered.photo = img_tk

    # Hàm nội bộ: Chuyển đổi ảnh sang ma trận mức xám
    def image_to_gray_matrix(self, image):
        """
        Chuyển ảnh PIL thành ma trận mức xám.
        Params:
            image: Ảnh PIL
        Returns:
            numpy.ndarray: Ma trận mức xám
        """
        gray_image = image.convert("L")
        return np.array(gray_image, dtype=np.uint8)

    # Hàm nội bộ: Chuyển đổi ma trận mức xám thành ảnh
    def gray_matrix_to_image(self, gray_matrix):
        """
        Chuyển đổi ma trận mức xám thành ảnh PIL.
        Params:
            gray_matrix: numpy.ndarray chứa dữ liệu mức xám
        Returns:
            Image: Ảnh PIL
        """
        if gray_matrix.dtype != np.uint8:
            gray_matrix = gray_matrix.astype(np.uint8)
        return Image.fromarray(gray_matrix, mode="L")

    def negative(self):
        if self.img is not None:
            gray_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(gray_matrix), len(gray_matrix[0])  
            for i in range(n):
                for j in range(m):
                    gray_matrix[i][j] = 255 - gray_matrix[i][j]
            self.display_result(gray_matrix)

    def apply_threshold(self):
        # Hiển thị thanh trượt khi chọn chức năng phân ngưỡng
        self.threshold_frame.grid(row=3, column=0, columnspan=2, pady=10)
        if self.img is not None:
            gray_matrix = self.image_to_gray_matrix(self.img)
            threshold_value = self.threshold_slider.get()
            n, m = len(gray_matrix), len(gray_matrix[0])  
            for i in range(n):
                for j in range(m):
                    if gray_matrix[i][j] >= threshold_value: gray_matrix[i][j] = 255 
                    else: gray_matrix[i][j] = 0
            self.display_result(gray_matrix)
    def linear_transformation(self):
        if self.img is not None:
            gray_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(gray_matrix), len(gray_matrix[0])  
            for i in range(n):
                for j in range(m):
                    val = 1.0 * gray_matrix[i][j] + 0
                    gray_matrix[i][j] = min(255, max(0, round(val)))
            self.display_result(gray_matrix)
    def histogram_equalization(self):
        if self.img is not None:
            image_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(image_matrix), len(image_matrix[0])
            x = 256  # Số mức xám (0-255)
            a = image_matrix
            cnt = [0] * x
            for i in range(n):
                for j in range(m): cnt[a[i][j]] += 1
            # Bước 2: Tính toán hàm phân phối tích lũy (CDF)
            s = [0] * x
            mp = [0] * x
            s[0] = (x - 1) * cnt[0] / (m * n)
            for i in range(1, x): s[i] = s[i - 1] + (x - 1) * cnt[i] / (m * n)
            for i in range(x): mp[i] = round(s[i])
            # Bước 3: Ánh xạ giá trị cũ sang giá trị mới
            result_matrix = np.zeros_like(a)
            for i in range(n):
                for j in range(m): result_matrix[i][j] = mp[a[i][j]]
            self.display_result(result_matrix)
    def apply_mean_filter(self):
        if self.img is not None:
            image_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(image_matrix), len(image_matrix[0])
            result_matrix = np.zeros_like(image_matrix)
            # Ma trận trọng số
            weights = [
                [1/16, 2/16, 1/16],
                [2/16, 4/16, 2/16],
                [1/16, 2/16, 1/16]
            ]
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    tong = 0
                    for x in range(3):
                        for y in range(3): tong += image_matrix[i + x - 1][j + y - 1] * weights[x][y]
                    result_matrix[i][j] = round(tong)
            # Xử lý biên (Cho giữ nguyên)
            for i in range(n):
                result_matrix[i][0] = image_matrix[i][0]
                result_matrix[i][m-1] = image_matrix[i][m-1]
            for j in range(m):
                result_matrix[0][j] = image_matrix[0][j]
                result_matrix[n-1][j] = image_matrix[n-1][j]
            self.display_result(result_matrix)
    def apply_median_filter(self):
        if self.img is not None:
            image_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(image_matrix), len(image_matrix[0])
            result_matrix = np.zeros_like(image_matrix)
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    tmp = []
                    for x in range(3):
                        for y in range(3): tmp.append(image_matrix[i + x - 1][j + y - 1])
                    tmp.sort()
                    result_matrix[i][j] = tmp[4]  
            # Xử lý biên (Cho giữ nguyên)
            for i in range(n):
                result_matrix[i][0] = image_matrix[i][0]
                result_matrix[i][m-1] = image_matrix[i][m-1]
            for j in range(m):
                result_matrix[0][j] = image_matrix[0][j]
                result_matrix[n-1][j] = image_matrix[n-1][j]
            self.display_result(result_matrix)
    def apply_error_diffusion(self):
        if self.img is not None:
            image_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(image_matrix), len(image_matrix[0])
            e = np.zeros_like(image_matrix)  # Lỗi (error) ban đầu
            b = np.zeros_like(image_matrix)  # Pixel đã phân ngưỡng
            u = np.zeros_like(image_matrix)  # Dự đoán
            for i in range(n):
                for j in range(m):
                    if j == 0: u[i][j] = image_matrix[i][j]
                    else:u[i][j] = image_matrix[i][j] - e[i][j - 1]
                    # So sánh ngưỡng để suy ra b. Cho ngưỡng = 127. Lớn hơn ngưỡng 255, ngược lại thì 0
                    if u[i][j] < 127: b[i][j] = 0
                    else: b[i][j] = 255
                    e[i][j] = b[i][j] - u[i][j]
                    # Khuếch tán lỗi (Floyd-Steinberg)
                    if j + 1 < m:  # Pixel bên phải
                        image_matrix[i][j + 1] = min(255, max(0, image_matrix[i][j + 1] + e[i][j] * 7 / 16))
                    if i + 1 < n and j - 1 >= 0:  # Pixel phía dưới bên trái
                        image_matrix[i + 1][j - 1] = min(255, max(0, image_matrix[i + 1][j - 1] + e[i][j] * 3 / 16))
                    if i + 1 < n:  # Pixel dưới
                        image_matrix[i + 1][j] = min(255, max(0, image_matrix[i + 1][j] + e[i][j] * 5 / 16))
                    if i + 1 < n and j + 1 < m:  # Pixel phía dưới bên phải
                        image_matrix[i + 1][j + 1] = min(255, max(0, image_matrix[i + 1][j + 1] + e[i][j] * 1 / 16))
            self.display_result(b)
    def apply_sobel(self):
        if self.img is not None:
            image_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(image_matrix), len(image_matrix[0]) 
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            result_matrix = np.zeros((n, m), dtype=np.uint8)
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    Gx, Gy = 0, 0
                    for x in range(3):
                        for y in range(3):
                            Gx += image_matrix[i + x - 1][j + y - 1] * sobel_x[x][y]
                            Gy += image_matrix[i + x - 1][j + y - 1] * sobel_y[x][y]
                    G = math.sqrt(Gx**2 + Gy**2)
                    result_matrix[i, j] = np.clip(int(G), 0, 255)
            self.display_result(result_matrix)   
    def apply_prewitt(self):
        if self.img is not None:
            image_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(image_matrix), len(image_matrix[0]) 
            prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            result_matrix = np.zeros((n, m), dtype=np.uint8)
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    Gx, Gy = 0, 0
                    for x in range(3):
                        for y in range(3):
                            Gx += image_matrix[i + x - 1][j + y - 1] * prewitt_x[x][y]
                            Gy += image_matrix[i + x - 1][j + y - 1] * prewitt_y[x][y]
                    G = math.sqrt(Gx**2 + Gy**2)
                    result_matrix[i, j] = np.clip(int(G), 0, 255)
            self.display_result(result_matrix) 
    def apply_laplace(self):
        if self.img is not None:
            image_matrix = self.image_to_gray_matrix(self.img) 
            n, m = len(image_matrix), len(image_matrix[0]) 
            laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            result_matrix = np.zeros_like(image_matrix)
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    val = 0
                    for x in range(3):
                        for y in range(3): val += image_matrix[i + x - 1][j + y - 1] * laplace_kernel[x][y]
                    result_matrix[i, j] = abs(val)
            self.display_result(result_matrix) 
    def apply_erosion(self):
        SE = [[0, 1, 0], [1, 1, 1], [0, 1, 0]] 
        if self.img is not None:
            gray_image = self.image_to_gray_matrix(self.img)
            n, m = len(gray_image), len(gray_image[0])  
            for i in range(n):
                for j in range(m): gray_image[i][j] = 1 if gray_image[i][j] >= 120 else 0
            n1 = len(SE)
            res = np.zeros_like(gray_image) 
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    check = 255  
                    for x in range(n1):
                        for y in range(n1):
                            if SE[x][y] == 1 and gray_image[i + x - 1][j + y - 1] == 0:
                                check = 0  
                                break
                        if check == 0: break
                    res[i][j] = check  
            self.display_result(res)
    def apply_dilation(self):
        SE = [[0, 1, 0], [1, 1, 1], [0, 1, 0]] 
        if self.img is not None:
            gray_image = self.image_to_gray_matrix(self.img)
            n, m = len(gray_image), len(gray_image[0])  
            for i in range(n):
                for j in range(m): gray_image[i][j] = 1 if gray_image[i][j] >= 80 else 0
            n1 = len(SE)
            res = np.zeros_like(gray_image) 
            for i in range(1, n - 1):
                for j in range(1, m - 1):
                    check = 0 
                    for x in range(n1):
                        for y in range(n1):
                            if SE[x][y] == 1 and gray_image[i + x - 1][j + y - 1] == 1:
                                check = 255  
                                break
                        if check == 255: break
                    res[i][j] = check  
            self.display_result(res)
    def apply_otsu(self):
        SE = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]  # SE là toán tử cấu trúc
        if self.img is not None:
            gray_image = self.image_to_gray_matrix(self.img)  # Chuyển ảnh sang ma trận xám
            n, m_img = len(gray_image), len(gray_image[0])  # Đổi tên m thành m_img để tránh xung đột
            cnt = np.zeros(256)  # Đếm tần suất các mức xám
            for i in range(n):
                for j in range(m_img):  # Sử dụng m_img thay vì m
                    cnt[gray_image[i][j]] += 1
            total_pixels = n * m_img  # Tính tổng số pixel
            p = np.zeros(256)  # Xác suất của các mức xám
            P = np.zeros(256)  # Xác suất tích lũy
            m_array = np.zeros(256)  # Mức xám tích lũy
            mG = 0  # Tổng mức xám của toàn ảnh
            for i in range(256):
                p[i] = cnt[i] / total_pixels
                if i == 0:
                    P[i] = p[i]
                    m_array[i] = i * p[i]
                else:
                    P[i] = P[i - 1] + p[i]
                    m_array[i] = m_array[i - 1] + i * p[i]
            mG = m_array[255]  # Tổng mức xám của toàn ảnh
            threshold = 0  # Ngưỡng tối ưu ban đầu
            max_variance = -1  # Phương sai tối đa ban đầu
            for i in range(1, 256):
                if P[i] == 0 or P[i] == 1: continue
                # Tính phương sai giữa hai lớp
                mu_t = mG * P[i] - m_array[i]
                sigma_b = (mu_t * mu_t) / (P[i] * (1 - P[i]))
                # Cập nhật ngưỡng nếu phương sai giữa các lớp lớn hơn phương sai tối đa hiện tại
                if sigma_b > max_variance:
                    max_variance = sigma_b
                    threshold = i
            for i in range(n):
                for j in range(m_img):
                    gray_image[i][j] = 255 if gray_image[i][j] >= threshold else 0
            self.display_result(gray_image)
# Chuyển ma trận thành ảnh
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
