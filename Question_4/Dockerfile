# Sử dụng hình ảnh Python chính thức
FROM python:3.9-slim

# Đặt thư mục làm việc trong container
WORKDIR /app

# Sao chép các file cần thiết vào container
COPY app.py classifier.py data_loader.py /app/
COPY svm_model.pkl /app/

# Cài đặt các gói cần thiết
RUN pip install flask Pillow numpy scikit-learn joblib opencv-python

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["python", "app.py"]
