import os

def rename_images_in_directory(directory, prefix):
    """
    重命名指定目录下的图片文件，格式为 xy，其中 x 是用户指定的前缀，y 是处理顺序编号。
    文件格式保持不变。

    :param directory: 包含图片的目录路径
    :param prefix: 用户指定的前缀
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"目录 {directory} 不存在，请检查路径！")
        return

    # 获取目录下的所有文件
    files = os.listdir(directory)

    # 过滤出图片文件（支持常见图片格式）
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

    # 按文件名排序（可选，确保处理顺序一致）
    image_files.sort()

    # 重命名图片文件
    for index, filename in enumerate(image_files, start=1):
        # 获取文件的扩展名
        file_extension = os.path.splitext(filename)[1]

        # 构造新的文件名
        new_filename = f"{prefix}{index}{file_extension}"

        # 构造完整的旧文件路径和新文件路径
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)

        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"已重命名：{filename} -> {new_filename}")

    print("所有图片文件已重命名完成！")

if __name__ == "__main__":
    # 用户指定目录和前缀
    directory_path = input("请输入包含图片的目录路径：")
    prefix_name = input("请输入图片文件的前缀：")

    # 调用函数重命名图片
    rename_images_in_directory(directory_path, prefix_name)