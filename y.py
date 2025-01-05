import cv2
import numpy as np
import os

# 定義顏色範圍和其顏色名稱
color_names = {
    'Red': ((0, 100, 100), (10, 255, 255)),
    'Orange': ((10, 150, 150), (25, 255, 255)),  
    'Yellow': ((25, 150, 150), (40, 255, 255)), 
    'Green': ((35, 100, 100), (85, 255, 255)),
    'Cyan': ((85, 100, 100), (95, 255, 255)), 
    'Blue': ((90, 100, 100), (140, 255, 255)),  
    'Purple': ((130, 100, 100), (150, 255, 255)), 
    'Pink': ((145, 80, 180), (170, 200, 255)),     
}
min_area = 7000  # 面積過小的區域可以忽略


image_folder = 'C://Users//user//images/'  
output_folder = 'C://Users//user//output/' 
os.makedirs(output_folder, exist_ok=True)

# 輸出圖片為800 * 800
standard_width = 800
standard_height = 800

# 取得取資料夾中的所有圖片
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]


for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"未讀取到圖片: {image_file}")
        continue

    # 調整圖片大小
    height, width = image.shape[:2]
    scale_width = standard_width / width
    scale_height = standard_height / height
    scale = min(scale_width, scale_height)  # 確保圖片不變形
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 填充到標準大小
    padded_image = np.zeros((standard_height, standard_width, 3), dtype=np.uint8)
    x_offset = (standard_width - new_width) // 2
    y_offset = (standard_height - new_height) // 2
    padded_image[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_image

    # 轉換顏色空間 BGR -> HSV
    hsv_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2HSV)
    
    # 記錄框的數量
    contour_count = 0  
    text_positions = []  # 記錄文字框位置避免重疊
    
   
    for color_name, (lower, upper) in color_names.items():
        # 生成單一顏色的遮罩
        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        # 找到顏色區域的輪廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 按輪廓面積排序（從大到小）
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # 只處理最大的幾個區域
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < min_area:  # 忽略過小的區域
                break

            # 繪製輪廓框
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(padded_image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 繪製邊框

            # 動態調整文字位置，避免與其他框重疊
            text_x = x
            text_y = y - 10 if y - 10 > 10 else y + h + 20

            # 檢查文字框是否會重疊，重疊則調整位置
            for (prev_x, prev_y) in text_positions:
                if abs(text_x - prev_x) < 100 and abs(text_y - prev_y) < 30:
                    text_y = prev_y + 30  # 避免重疊，將其移到新的位置
                    break

            # 添加背景矩形
            text_size = cv2.getTextSize(color_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(padded_image, (text_x, text_y - text_size[1] - 5),
                        (text_x + text_size[0] + 5, text_y + 5), (0, 255, 0), -1)

            # 在背景矩形上添加文字標籤
            cv2.putText(padded_image, color_name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 2)

            contour_count += 1
            text_positions.append((text_x, text_y))  # 記錄文字框位置

    # 6. 儲存處理後的圖片
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, padded_image)
    print(f"已處理並儲存: {output_path}")

print("所有圖片處理完成！")
