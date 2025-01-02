import os
import re
import shutil

def delete_files_and_folders():
    # 将工作目录切换为脚本所在的目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    print(f"当前工作目录: {script_dir}")

    # 正则表达式匹配 "exp_data-" 开头和 9 个连续数字结尾
    folder_pattern = re.compile(r"^exp_data-\d{10}$")

    # 遍历当前目录的所有文件和文件夹
    for item in os.listdir(script_dir):
        item_path = os.path.join(script_dir, item)

        # 删除符合条件的文件
        if os.path.isfile(item_path) and item.startswith("output") and item.endswith(".log"):
            try:
                os.remove(item_path)
                print(f"已删除文件: {item_path}")
            except Exception as e:
                print(f"无法删除文件 {item_path}: {e}")

        # 删除符合条件的文件夹
        elif os.path.isdir(item_path) and folder_pattern.match(item):
            try:
                shutil.rmtree(item_path)
                print(f"已删除文件夹: {item_path}")
            except Exception as e:
                print(f"无法删除文件夹 {item_path}: {e}")

if __name__ == "__main__":
    delete_files_and_folders()