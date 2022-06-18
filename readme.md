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
## 1. Hàm khởi tạo ```Centroid``` ngẫu nhiên
---
```py
def init_centroids_kmean(img_1d,k_clusters,init_centroids):
    if init_centroids == 'random':
        return np.random.randint(0,255, size=(k_clusters, len(img_1d[0])))
    elif init_centroids == 'in_pixels':
        return img_1d[np.random.choice(img_1d.shape[0], size=k_clusters, replace=False)]
    else:
        raise ValueError('init_centroids must be either "random" or "in_pixels"')
```
- Đầu vào
    - *img_1d (np.array)*: Nội dung của bức ảnh sau khi chuyển đổi thành ma trận và reshape(cao*rộng,3)
    - *k_clusters (int):* `Số lượng` màu muốn nén
    - *init_centroids (str)*: Chuỗi *random* hoặc *in_pixel* để khởi tạo
- Đầu ra
    - Trả về một ma trận có các điểm ảnh từ [0,255] nếu muốn khởi tạo ```Centroid``` bằng *random*
    - Trả về một ma trận có các điểm ảnh lấy từ ảnh nếu muốn khởi tạo ```Centroid``` bằng *in_pixel*
### ***Ý tưởng thực hiện***
* Ở đây ta sử dụng `np.random.randint` và `np.random.choice`
    * `replace=False` để không trùng giá trị **Centroid**

## 2. Hàm tìm ra danh sách lưu các vị trí gần với `Centroid`
---
```py
def get_label(img_1d, centroid):
    list_distance=[]
    for i in range(0,len(centroid)):
        distance = np.linalg.norm(np.subtract(img_1d ,centroid[i]), axis=1) #tính khoảng cách giữa ảnh và từng centroid
        list_distance.append(distance) #lưu khoảng cách vào mảng
    label = np.argmin(list_distance, axis=0) #lấy index của khoảng cách nhỏ nhất
    return label
```
- Đầu vào:
    - *img_1d (np.array)*: Nội dung của bức ảnh sau khi chuyển đổi thành ma trận và reshape(cao*rộng,3)
    - *centroid (np.array)*: Danh sách màu `Centroid`
- Đầu ra:
    - Trả về một danh sách chứa vị trí các màu gần `Centroid`
### ***Ý tưởng thực hiện***
+ Đầu tiên sẽ tạo ra một danh sách, sau đó tiến hành chạy một vòng lặp for từ 0 đến `len(centroid)`, `len(centroid)` ở đây chính là `số lượng` màu trung tâm. Ta lần lượt tính khoảng cách toàn bộ ma trận hình ảnh với từng hàng trong ma trận của `Centroid`.
    + Sử dụng hàm `np.subtract` dùng để trừ 2 vector
    + Sử dụng hàm `np.linalg.norm` để tính khoảng khách, và *axis=1* nghĩa là được thực hiện trên hàng
    + Lưu khoảng cách vừa tính vào danh sách.
    + Sau khi lưu tất cả khoảng cách, ta sẽ tìm ra các vị trí có khoảng cách gần với `Centroid` bằng cách sử dụng hàm `np.argmin`, *axis=0* nghĩa là được thực hiện trên từng cột. Vì khi lưu danh sách nó sẽ lưu thành từng hàng nên ta sẽ tìm vị trí khoảng cách ngắn nhất trên từng cột.
## 3. Hàm tính `Centroid` mới
---
```py
def get_centroid(img_1d, label, k_clusters):
    centroid=np.zeros((k_clusters, len(img_1d[0])))
    for i in range (k_clusters):
        k=img_1d[label==i] # chọn các vector gần centroid gán lại cho k
        if(len(k)!=0): # kiểm tra xem k có rỗng không
            centroid[i]=np.mean(k,axis=0) # tính trung bình cộng các vector
    return centroid 
```
- Đầu vào:
    - *img_1d (np.array)*: Nội dung của bức ảnh sau khi chuyển đổi thành ma trận và reshape(cao*rộng,3).
    - *label (np.array)*: Danh sách lưu trữ các giá trị `{0,1,2,…k_cluster}` tương ứng với `Centroid` mà điểm thuộc về.
    - *k_clusters (int):* `Số lượng` màu muốn nén.
- Đầu ra:
    - Trả về `Centroid` mới sau khi tính trung bình cộng các điểm gần trung tâm
### ***Ý tưởng thực hiện***
+ Đầu tiên ta sẽ khởi tạo `Centroid` bằng `np.zeros` là tạo ra ma trận `0` với k_clusters dòng và `len(img_1d[0])` mặc định là 3
    + Chạy 1 vòng lặp for từ 0 đến k_clusters
        + `k=img_1d[label==i]` nghĩa là chọn tất cả các vector gần với `Centroid[i]` và gán vào biến `k`
        + `if(len(k)!=0)` Kiểm tra xem độ dài có `khác 0` không để xét trường hợp không có vector nào gần với `Centroid[i]`
        + Tính trung bình cộng các vector bằng `np.mean`, *axis=1* nghĩa là tính theo từng cột để trả ra vector gán vào `Centroid[i]`
        + Trả về một centroid mới
## 4. Hàm `Kmeans` dùng để nén ảnh
---
```py
def kmeans(img_1d, k_clusters, max_iter, init_centroids='random'):
    flag=0
    centroid=init_centroids_kmean(img_1d,k_clusters,init_centroids) #khởi tạo centroid
    for i in range(max_iter):
        label = get_label(img_1d, centroid) #lấy label của ảnh
        new_centroid = get_centroid(img_1d, label, k_clusters) #cập nhật centroid
        if np.array_equal(centroid,new_centroid): # neu khong co su thay đổi về centroid thì dừng vòng lặp
            flag=1
            break
        centroid = new_centroid #cập nhật centroid mới
    if(flag==0): # nếu chạy hết vòng lăp thì ta tiến cập nhật lại label
        label=get_label(img_1d,centroid)
    return label, centroid
```
- Đầu vào:
    - *img_1d (np.array)*: Nội dung của bức ảnh sau khi chuyển đổi thành ma trận và reshape(cao*rộng,3).
    - *k_clusters (int):* `Số lượng` màu muốn nén.
    - *max_iter (int)*: `số lượng` vòng lặp
    - *init_centroids (str)*: Chuỗi *random* hoặc *in_pixel* để khởi tạo
- Đầu ra:
    - *label (np.array)*: Danh sách lưu trữ các giá trị `{0,1,2,…k_cluster}` tương ứng với `Centroid` mà điểm thuộc về.
    - *centroid (np.array)*: Danh sách màu `Centroid`
### ***Ý tưởng thực hiện***
+ Tạo 1 biến `flag=0` để làm cờ
+ Khởi tạo `Centroid` bằng cách gọi lại hàm `init_centroids_kmean(img_1d,k_clusters,init_centroids)`
+ Chạy vòng lặp for từ 0 đến max_iter
    + Gọi hàm `get_label(img_1d, centroid)` gán vào `label`
    + Gọi hàm `get_centroid(img_1d, label, k_clusters)` gán vào `new_centroid`
    + sử dụng `np.array_equal` kiểm tra xem giữa `centroid` và `new_centroid` có thay đổi không. Nếu không thay đổi thì gán `flag=1` và dừng vòng lặp.
    + Gán `centroid ` bằng `new_centroid`
    + Cứ lặp đến khi có điều kiện dừng hoặc hết vòng lặp
    + Nếu chạy đủ vòng lặp thì ta gán lại cho label 1 lần nữa bằng `get_label(img_1d,centroid)`. Vì khi dừng vòng lặp `centroid` mới vừa cập nhật nên ta phải gọi thêm 1 lần nữa để truyền đúng `centroid` mà ta mới cập nhật
    + Trả về `label` và `centroid`
## 5. Hàm `main`
```py
if __name__ == '__main__':
    img_file=input("Enter the image file name: ")
    img = Image.open(img_file)
    img_1d = np.array(img)
    img_shape=img.shape
    img_1d = img_1d.reshape((img_1d.shape[0] * img_1d.shape[1], img_1d.shape[2]))
    k_clusters = int(input("Enter the number of clusters: "))
    max_iter = int(input("Enter the maximum number of iterations: "))
    init_centroids = input("Enter the initial centroids: ")
    print("K-means clustering...")
    label, centroid = kmeans(img_1d,k_clusters,max_iter, "random")
    result = centroid[label].astype(np.uint8) #thay thế điểm ảnh bằng centroid của nó
    result = result.reshape(img_shape) #reshape result về ảnh
    plt.imshow(result) #hiển thị ảnh
    plt.show()
    file_name=input("Enter the file name want to save [filename.(jpg/png/pdf)]: ")
    plt.imsave(file_name,result)
```
### ***Ý tưởng thực hiện***
- Nhập tên `file ảnh `cần nén
- Dùng `Image.open ` để mở file ảnh
- Chuyển đổi ảnh lưu trữ với dạng mảng kiểu numpy
- Lúc này nội dung của mảng có shape(cao,rộng,3) ta tiến hành reshape(cao*rộng,3) để dễ xử lý
- Nhập số lượng màu cần nén
- Nhập số vòng lặp
- Nhập kiểu muốn khởi tạo `Centroid` *random* hoặc *in_pixels*
- Gọi hàm `Kmeans`
- Sau khi trả về `label` và `centroid` từ hàm `Kmeans`. Ta tiến hành thay thế các điểm ảnh bằng `Centroid` của nó
- Chuyển về ảnh gốc bằng cách xài `reshape`
- Sử dụng `plt.show` và `plt.imshow()` để hiển thị ảnh
- Lưu ảnh bằng `plt.imsave`
# Hình ảnh kết quả
<div class="name" style="margin-top: 100px;">
    <p style="font-size: 30px;margin-left: 120px;margin-top: -70px; float: left">Ảnh gốc</p>
    <div class="hihi" style="postion: relative;">
        <p style="font-size: 30px;margin-top: -70px;margin-left:380px; float:left;">K=3</p>
        <p style="font-size: 30px;margin-top: -70px;margin-left:540px; float:left;">K=5</p>
        <p style="font-size: 30px;margin-top: -70px;margin-left:700px; float:left;">K=7</p>
        <p style="font-size: 30px;margin-top: 100px;margin-left:950px; float:left;">Random</p>
        <p style="font-size: 30px;margin-top: 200px;margin-left:950px; float:left;">In_pixels</p>
    </div>
    <div class="ccc">
        <div class="h1" style="position: absolute;">
        <img style="float: left;" src="https://i.imgur.com/EYzQgRR.jpeg" alt="jisoo" width="325"/> 
            <img align="top"  src="https://i.imgur.com/Ab70eFf.jpg" style="margin: 0px 0px 30px 10px;" alt="js1" width="150"/>
            <img align="top" src="https://i.imgur.com/iliTstD.jpg" style="margin: 0px 0px 30px 10px;" alt="js1" width="150"/>
            <img align="top" src="https://i.imgur.com/vW4our0.jpg" style="margin: 0px 0px 30px 10px; display: inline-block;" alt="js1" width="150"/> 
            <div class="cc">
                <img src="https://i.imgur.com/aN1ZTGy.jpg" alt="js4" style=" margin-left: 10px;" width="150"/>
                <img  src="https://i.imgur.com/yNp8rrQ.jpg" alt="js5" style=" margin-left: 10px;" width="150"/>
                <img  src="https://i.imgur.com/lsHGrps.jpg" alt="js6" style="margin-left: 10px;display: inline-block;" width="150"/>
            </div>
        </div>
    </div>
</div>




