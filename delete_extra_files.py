import os
import random

# 指定文件夹路径
folder_path = 'samples_deepcad'

# 获取所有step文件的列表
step_files = [f for f in os.listdir(folder_path) if f.endswith('.step')]

# 随机选择15个step文件
selected_step_files = random.sample(step_files, 318)

# 删除选定的step文件及其对应的stl文件
for step_file in selected_step_files:
    # 找到对应的stl文件
    stl_file = step_file.replace('.step', '.stl')

    # 构造文件路径
    step_file_path = os.path.join(folder_path, step_file)
    stl_file_path = os.path.join(folder_path, stl_file)

    # 删除文件
    os.remove(step_file_path)
    os.remove(stl_file_path)
    print(f"Deleted: {step_file} and {stl_file}")

print("318对文件已删除。")
