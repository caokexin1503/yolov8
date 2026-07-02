import os

from PIL import Image

# ==================== 路径设置（请确认） ====================
LABEL_DIR = r"C:\Users\caoke\Desktop\ultralytics-main\datasets\data\labels\val"
IMAGE_DIR = r"C:\Users\caoke\Desktop\ultralytics-main\datasets\data\images\val"
# ==========================================================


def convert_single_file(txt_path, img_w, img_h):
    with open(txt_path) as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < 8:
            continue

        xmin = float(parts[0])
        ymin = float(parts[1])
        width = float(parts[2])
        height = float(parts[3])
        score = int(float(parts[4]))
        class_id = int(float(parts[5]))
        # truncation = int(float(parts[6]))   # 标准YOLO不需要，若需保留见说明
        # occlusion  = int(float(parts[7]))

        # 跳过忽略区域
        if score == 0:
            continue

        # 计算归一化YOLO坐标
        x_center = (xmin + width / 2.0) / img_w
        y_center = (ymin + height / 2.0) / img_h
        w_norm = width / img_w
        h_norm = height / img_h

        # 防止浮点误差越界
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    # 覆盖原文件
    with open(txt_path, "w") as f:
        f.write("\n".join(new_lines))


def main():
    if not os.path.isdir(LABEL_DIR):
        print(f"错误：标签文件夹不存在 {LABEL_DIR}")
        return
    if not os.path.isdir(IMAGE_DIR):
        print(f"错误：图片文件夹不存在 {IMAGE_DIR}")
        return

    # 支持的图片后缀
    img_exts = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    txt_files = [f for f in os.listdir(LABEL_DIR) if f.endswith(".txt")]

    for filename in txt_files:
        base = os.path.splitext(filename)[0]
        img_path = None
        for ext in img_exts:
            candidate = os.path.join(IMAGE_DIR, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f"警告：找不到对应图片 {base}.* ，跳过")
            continue

        with Image.open(img_path) as img:
            w, h = img.size

        txt_path = os.path.join(LABEL_DIR, filename)
        convert_single_file(txt_path, w, h)
        print(f"已转换: {filename}")

    print("全部转换完成！")


if __name__ == "__main__":
    main()
