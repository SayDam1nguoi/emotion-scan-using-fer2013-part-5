# Dự án quét cảm xúc qua camera trực tiếp 
# Các chức năng quét hiện đang có
- Quét trực tiếp qua camera
- Quét qua 1 vùng được chọn (dùng cho muốn quét người khác )
- Quét khuôn mặt qua video
# Các chức năng 
- Ra kết quả và tỷ lệ cảm xúc chiếm nhiều nhất -> ra được cần cải thiện cảm xúc nào
- Quét bối cảnh, trang phục -> cải thiện trang phục (beta)
- Quét tỷ lệ tập trung -> tránh việc khi đang trò chuyện thì mắt người kia không tập trung vào màn hình đối phương (beta)
# Các thư viện cần cài
pip install tensorflow opencv-python mtcnn numpy scikit-learn matplotlib Pillow mss
# Database cần phải có
fer2013.csv
