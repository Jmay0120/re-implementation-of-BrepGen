import os

# 设置目标文件夹路径
folder_path = 'samples_abc'

# 获取所有 .step 和 .stl 文件的名称（不包括扩展名）
step_files = {os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.step')}
stl_files = {os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.stl')}

# 找出没有对应文件的 .step 文件
step_without_stl = step_files - stl_files
# 找出没有对应文件的 .stl 文件
stl_without_step = stl_files - step_files

# 打印没有对应文件的 .step 文件
if step_without_stl:
    print("The following .step files do not have corresponding .stl files:")
    for file in step_without_stl:
        file_path = os.path.join(folder_path, f"{file}.step")
        os.remove(file_path)
        print(f"Deleted {file}.step")
else:
    print("All .step files have corresponding .stl files.")

# 打印没有对应文件的 .stl 文件
if stl_without_step:
    print("The following .stl files do not have corresponding .step files:")
    for file in stl_without_step:
        file_path = os.path.join(folder_path, f"{file}.stl")
        os.remove(file_path)
        print(f"Deleted {file}.stl")
else:
    print("All .stl files have corresponding .step files.")
