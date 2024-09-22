import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights

# Khởi tạo mô hình ResNet50
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model.eval()

def adjust_feature_size(features, target_size=1000):
    if len(features) > target_size:
        features = features[:target_size]
    elif len(features) < target_size:
        features = np.pad(features, (0, target_size - len(features)), mode='constant')
    return features

def extract_features(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image_tensor)

    features_np = features.flatten().numpy()
    features_np = adjust_feature_size(features_np)
    return features_np

def save_features_to_csv(image_path, features, output_folder):
    """
    Lưu đặc trưng vào file CSV với đường dẫn ảnh trong cột đầu tiên.
    """
    try:
        # Tạo tên file đầu ra
        feature_filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(image_path))[0]}_features_with_paths.csv")
        
        # Chuẩn bị dữ liệu để lưu vào CSV
        data = [image_path] + features.tolist()  # Đường dẫn ảnh ở cột đầu tiên
        
        # Lưu vào file CSV
        df = pd.DataFrame([data])
        df.to_csv(feature_filename, index=False, header=False)
        
        print(f"Đặc trưng của {os.path.basename(image_path)} đã được lưu tại: {feature_filename}")
    
    except Exception as e:
        print(f"Đã xảy ra lỗi khi lưu đặc trưng: {e}")

def process_folder_and_save_features(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if os.path.isfile(image_path):
            try:
                # Trích xuất đặc trưng từ ảnh
                image = Image.open(image_path).convert('RGB')
                features = extract_features(image)
                
                # Lưu đặc trưng vào file CSV
                save_features_to_csv(image_path, features, output_folder)
            
            except Exception as e:
                print(f"Đã xảy ra lỗi khi xử lý ảnh {image_path}: {e}")

# Đường dẫn đến thư mục chứa ảnh
folder_path = r"D:\LUANVAN\resnet50\tach_anh"
output_folder = r"D:\LUANVAN\resnet50\dactrung"

# Xử lý toàn bộ ảnh trong thư mục và lưu đặc trưng
process_folder_and_save_features(folder_path, output_folder)
