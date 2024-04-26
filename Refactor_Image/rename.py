import os

path = "H:\\DAT\\HCSDL_DPT\\Data\\"
name = "picture"
extension = ".jpg"
count = 1
for folder in os.listdir(path):
    folder_path = os.path.join(path, folder)
    if os.path.isdir(folder_path):
        print(folder_path)
        for file in os.listdir(folder_path):
            open_path = os.path.join(folder_path, file)
            save_name = name + str(count) + extension
            save_path = os.path.join(path, save_name)
            os.rename(open_path, save_path)
            count += 1

