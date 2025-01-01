from flask import Flask, request, render_template, send_from_directory
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
# Load the lane detection PyTorch model
# model = torch.load('web_lane2/model/model_521.pt')
# model.eval()  # Set the model to evaluation mode



# @app.route('/detect', methods=['POST'])
# def detect():
#     try:
#         # Nhận file từ request
#         file = request.files['file']
#         img = Image.open(file.stream).convert("RGB")  # Đảm bảo ảnh ở định dạng RGB
#         img_array = np.array(img)  # Chuyển ảnh sang numpy array
        
#         # Tiền xử lý ảnh (nếu cần thiết: resize, normalize, convert to tensor)
#         input_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)  # Chuyển từ RGB sang BGR nếu cần thiết
#         input_img = cv2.resize(input_img, (640, 480))  # Resize nếu model yêu cầu kích thước cố định
#         input_tensor = torch.from_numpy(input_img).float().unsqueeze(0)  # Convert to tensor và thêm batch dimension
        
#         # Normalize ảnh nếu cần thiết
#         input_tensor /= 255.0
        
#         # Chạy dự đoán
#         with torch.no_grad():  # Không tính gradient trong quá trình inference
#             outputs = model(input_tensor)  # Dự đoán với mô hình lane detection
            
#         # Giả sử output là binary mask hoặc points xác định lanes (có thể thay đổi tùy thuộc vào mô hình của bạn)
#         # Chuyển output về dạng ảnh hoặc điểm cần vẽ (phụ thuộc vào kiểu output của model)
#         # Ví dụ, output là binary mask (1 cho lane, 0 cho background)
#         lane_mask = outputs.squeeze().cpu().numpy()  # Chuyển sang numpy array

#         # Vẽ lanes lên ảnh
#         result_img = img_array.copy()
#         result_img[lane_mask == 1] = [0, 0, 255]  # Gán màu đỏ cho vùng lane (BGR)

#         # Chuyển ảnh numpy array thành BytesIO để trả về
#         result_img = Image.fromarray(result_img)
#         img_io = io.BytesIO()
#         result_img.save(img_io, 'JPEG', quality=85)
#         img_io.seek(0)
#         return send_file(img_io, mimetype='image/jpeg')
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        # Lưu ảnh tải lên
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Xử lý ảnh (ví dụ đơn giản là sao chép sang thư mục kết quả)
        result_path = os.path.join(app.config['RESULT_FOLDER'], 'r' + file.filename)
        os.system(f'cp {filepath} {result_path}')

        # Hiển thị trang kết quả
        return render_template('index.html', uploaded_file=file.filename, result_file='r' + file.filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
