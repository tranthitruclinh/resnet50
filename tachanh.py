import fitz  # type: ignore # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_folder):
    # Lấy tên file PDF mà không cần đường dẫn và phần mở rộng
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Mở file PDF
    pdf_document = fitz.open(pdf_path)
    print(f"[debug] pages: {len(pdf_document)}")
    for page_number in range(len(pdf_document)):
        # Lấy từng trang trong PDF
        page = pdf_document.load_page(page_number)
        
        # Trích xuất các ảnh trên trang
        image_list = page.get_images(full=True)
        
        for image_index, image in enumerate(image_list):
            # Lấy mã số hình ảnh
            xref = image[0]
            
            # Trích xuất hình ảnh
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            # Đặt tên và lưu hình ảnh (bổ sung tên file PDF)
            image_filename = f"{output_folder}/{pdf_name}_page_{page_number + 1}_image_{image_index + 1}.{image_ext}"
            with open(image_filename, "wb") as image_file:
                image_file.write(image_bytes)
                
            print(f"Đã lưu hình ảnh: {image_filename}")
            print(f"[debug] page save: {page_number}")

    # Đóng file PDF
    pdf_document.close()

def extract_images_from_folder(folder_path, output_folder):
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_folder, exist_ok=True)
    
    # Duyệt qua tất cả các file trong thư mục
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Đang xử lý file PDF: {pdf_path}")
            extract_images_from_pdf(pdf_path, output_folder)

# Sử dụng hàm extract_images_from_folder
input_folder = r"D:\LUANVAN\resnet50\EN2015-2023-20240809T053334Z-001\EN2015-2023\2022\Vol 14-3"
output_folder = r"D:\LUANVAN\resnet50\tach_anh"
extract_images_from_folder(input_folder, output_folder)
