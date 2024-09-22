import glob
import numpy as np
import pandas as pd
import os
from PIL import Image
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import ResNet50_Weights

# Tải mô hình ResNet50 đã huấn luyện với trọng số mới nhất
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.eval()

def load_feature_from_csv(csv_path):
    """Đọc file CSV và trả về các đặc trưng và đường dẫn ảnh"""
    data = pd.read_csv(csv_path, header=None)
    image_paths = data.iloc[:, 0].astype(str).values
    vectors = data.iloc[:, 1:].values
    return image_paths, vectors

def extract_features(image):
    """Trích xuất đặc trưng từ ảnh tải lên"""
    # Chuyển đổi ảnh thành tensor và chuẩn hóa
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Thêm batch dimension
    
    with torch.no_grad():
        # Lấy đặc trưng từ tầng cuối cùng của mô hình
        features = model(input_batch)
    
    # Chuyển đổi tensor thành numpy array và giảm kích thước
    features = features.numpy().flatten()
    return features

def resize_vector(vector, new_size):
    """Thay đổi kích thước của vector để khớp với kích thước mới"""
    if len(vector) > new_size:
        return vector[:new_size]
    else:
        return np.pad(vector, (0, new_size - len(vector)), 'constant')

def find_most_similar_images(target_vector, target_image_path, csv_directory, top_n=5):
    """Tìm nhiều ảnh tương đồng nhất từ các file CSV trong thư mục, trừ ảnh mục tiêu"""
    correlations = []
    file_paths = glob.glob(os.path.join(csv_directory, '*.csv'))
    
    for file_path in file_paths:
        try:
            image_paths, vectors = load_feature_from_csv(file_path)
            vector_size = target_vector.shape[0]
            for idx, vector in enumerate(vectors):
                # Điều chỉnh kích thước vector nếu cần
                vector = resize_vector(vector, vector_size)
                # So sánh ảnh tải lên với các vector trong CSV
                similarity = cosine_similarity([target_vector], [vector])[0, 0]
                image_path = str(image_paths[idx]).strip()
                if os.path.isfile(image_path) and image_path != target_image_path:
                    correlations.append((image_path, similarity))
                elif image_path == target_image_path:
                    st.info(f"Ảnh '{image_path}' đã bị loại bỏ vì trùng với ảnh mục tiêu.")
                else:
                    st.warning(f"Đường dẫn ảnh '{image_path}' không hợp lệ hoặc không tồn tại.")
        except Exception as e:
            st.error(f"Không thể xử lý file {file_path}. Lỗi: {e}")

    if correlations:
        # Sắp xếp các ảnh theo độ tương đồng giảm dần
        correlations.sort(key=lambda x: x[1], reverse=True)
        # Trả về top N ảnh có độ tương đồng cao nhất
        return correlations[:top_n]
    
    return []

# Tạo giao diện người dùng với Streamlit
st.title("Tìm Ảnh Tương Đồng")

# Tải ảnh lên
uploaded_file = st.file_uploader("Tải lên một tấm ảnh", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh đã tải lên
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

    # Lưu ảnh mục tiêu tạm thời để so sánh
    target_image_path = uploaded_file.name  # Lấy tên file của ảnh mục tiêu

    st.write("Đang xử lý tìm ảnh tương đồng")
    
    # Trích xuất đặc trưng từ ảnh tải lên
    target_vector = extract_features(image)
    
    # Thư mục chứa các file CSV đặc trưng
    csv_directory = r"D:\LUANVAN\resnet50\dactrung"
    
    # Tìm nhiều ảnh tương đồng nhất, không trùng với ảnh mục tiêu
    top_similar_images = find_most_similar_images(target_vector, target_image_path, csv_directory, top_n=5)
    
    if top_similar_images:
        st.write(f"Tìm thấy {len(top_similar_images)} ảnh tương đồng nhất:")
        for i, (image_path, similarity) in enumerate(top_similar_images):
            col1, col2 = st.columns(2)

            with col1:
                st.image(image, caption="Ảnh đã tải lên", use_column_width=True)

            with col2:
                similar_image = Image.open(image_path)
                st.image(similar_image, caption=f"Ảnh #{i+1} (Độ tương đồng: {similarity:.4f})", use_column_width=True)
    else:
        st.write("Không tìm thấy ảnh nào để so sánh.")
