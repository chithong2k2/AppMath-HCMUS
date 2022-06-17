# **Thông tin cá nhân**
Họ tên: Dương Chí Thông

MSSV: 20127634

Lớp: 20CLC08

# **Ý tưởng thực hiện**
* Gồm 5 bước:
1. Chọn K điểm bất kỳ làm các center ban đầu.
2.	Phân mỗi điểm dữ liệu vào cluster có center gần nó nhất.
3.	Nếu việc gán dữ liệu vào từng cluster ở bước 2 không thay đổi so với vòng lặp trước nó thì ta dừng thuật toán.
4.	Cập nhật center cho từng cluster bằng cách lấy trung bình cộng của tất các các điểm dữ liệu đã được gán vào cluster đó sau bước 2.
5.	Quay lại bước 2.

# **Mô tả hàm**
1. Hàm khởi tạo ```Centroid``` ngẫu nhiên
```php
def init_centroids_kmean(img_1d,k_clusters,init_centroids):
    if init_centroids == 'random':
        return np.random.randint(0,255, size=(k_clusters, len(img_1d[0])))
    elif init_centroids == 'in_pixels':
        return img_1d[np.random.choice(img_1d.shape[0], size=k_clusters, replace=False)]
    else:
        raise ValueError('init_centroids must be either "random" or "in_pixels"')
```
- Đầu vào
    - img_1d (np.array): Nội dung của bức ảnh sau khi chuyển đổi thành ma trận và reshape(cao*rộng,3)
    - k_clusters (int): Số lượng màu muốn nén
    - init_centroids (str): Chuỗi *random* hoặc *in_pixels*
- Đầu ra
    - Trả về một ma trận có các điểm ảnh từ [0,255] nếu muốn khởi tạo ```Centroid``` bằng *random*
    - Trả về một ma trận có các điểm ảnh lấy từ ảnh nếu muốn khởi tạo ```Centroid``` bằng *in_pixels*
* Ở đây ta sử dụng `np.random.randint` và `np.random.choice`
    * `replace=False` để không trùng giá trị **Centroid**

2. Hàm tìm ra danh sách lưu các vị trí gần với `Centroid`
```php
def get_label(img_1d, centroid):
    list_distance=[]
    for i in range(0,len(centroid)):
        distance = np.linalg.norm(np.subtract(img_1d ,centroid[i]), axis=1) #tính khoảng cách giữa ảnh và từng centroid
        list_distance.append(distance) #lưu khoảng cách vào mảng
    label = np.argmin(list_distance, axis=0) #lấy index của khoảng cách nhỏ nhất
    return label
```
- Đầu vào:
    - img_1d (np.array): Nội dung của bức ảnh sau khi chuyển đổi thành ma trận và reshape(cao*rộng,3)
    - centroid (np.array): Danh sách màu `Centroid`
- Đầu ra:
    - Trả về một danh sách chứa vị trí các màu gần `Centroid`
+ Đầu tiên sẽ tạo ra một danh sách, sau đó tiến hành chạy một vòng lặp for từ 0 đến `len(centroid)`, `len(centroid)` ở đây chính là số lượng màu trung tâm. Ta lần lượt tính khoảng cách toàn bộ ma trận hình ảnh với từng hàng trong ma trận của `Centroid`.
    + Sử dụng hàm `np.subtract` dùng để trừ 2 vector
    + Sử dụng hàm `np.linalg.norm` để tính khoảng khách, và *axis=1* nghĩa là được thực hiện trên hàng
    + Lưu khoảng cách vừa tính vào danh sách.
    + Sau khi lưu tất cả khoảng cách, ta sẽ tìm ra các vị trí có khoảng cách gần với `Centroid` bằng cách sử dụng hàm `np.argmin`, *axis=0* nghĩa là được thực hiện trên từng cột. Vì khi lưu danh sách nó sẽ lưu thành từng hàng nên ta sẽ tìm vị trí khoảng cách ngắn nhất trên từng cột.
3. Hàm tính `Centroid`
```php
def get_centroid(img_1d, label, k_clusters):
    centroid=np.zeros((k_clusters, len(img_1d[0])))
    for i in range (k_clusters):
        k=img_1d[label==i]
        if(len(k)!=0):
            centroid[i]=np.mean(k,axis=0)
    return centroid 
```
- Đầu vào:
    - img_1d (np.array): Nội dung của bức ảnh sau khi chuyển đổi thành ma trận và reshape(cao*rộng,3).
    - label (np.array): Danh sách lưu trữ các giá trị `{0,1,2,…k_cluster}` tương ứng với `Centroid` mà điểm thuộc về.
    - k_clusters (int): Số lượng màu muốn nén.
- Đầu ra:
    - Trả về `Centroid` mới sau khi tính trung bình cộng các điểm gần trung tâm
+ Đầu tiên ta sẽ khởi tạo `Centroid` bằng `np.zeros` là tạo ra ma trận `0` với k_clusters dòng và `len(img_1d[0])` mặc định là 3
    + Chạy 1 vòng lặp for từ 0 đến k_clusters
        +
