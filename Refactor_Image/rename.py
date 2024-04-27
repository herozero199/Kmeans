import os

path = "H:\\DAT\\HCSDL_DPT\\Data\\"
name = "pic"
extension = ".jpg"
count = 1
for file in os.listdir(path):
    # file_path = os.path.join(path, file)
    open_path = os.path.join(path, file)
    save_name = name + str(count) + extension
    save_path = os.path.join(path, save_name)
    os.rename(open_path, save_path)
    count += 1

