
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np


def int_to_binary_array(num):
    # 将整数转为二进制字符串，填充到 4 位，然后转换为 ndarray
    return np.array(list(f"{num:04b}"), dtype=int)


def draw_picture(x_list: list, y_list: list, x_title: str, y_title: str, save_path: str):
    plt.clf()
    plt.figure(1)
    plt.plot(x_list, y_list)
    plt.xlabel(x_title)
    plt.ylabel(y_title)
    plt.grid()
    plt.savefig(save_path)


def clear_directory(directory_path):
    # 检查目录是否存在
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        return

    # 检查目录是否为空
    if not os.listdir(directory_path):
        print(f"The directory {directory_path} is already empty.")
        return

    # 清空目录中的内容
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # 删除文件或符号链接
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # 删除目录及其内容
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    print(f"The directory {directory_path} has been cleared.")


def make_dir_if_not_exist(directory_path):
    clear_directory(directory_path)
    if not os.path.exists(directory_path):
        print(f"The directory {directory_path} does not exist.")
        os.makedirs(directory_path)
