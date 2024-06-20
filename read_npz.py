import numpy as np
import pandas as pd


def read_npz_file(file_path):
    data = np.load(file_path)

    # .npz files are numpy archives, they can contain multiple arrays.
    # To access the arrays you need to use the key for each array.
    # For demonstration, we print the keys (array names).
    print("Keys: ", data.files)

    # Now you can access individual array in the npz file by these keys
    for key in data.files:
        print(f"{key}(f'{data[key].shape}'):\n", data[key])

# def npz_to_csv(file_path, csv_path_prefix):
#     data = np.load(file_path)
#
#     for key in data.files:
#         df = pd.DataFrame(data[key])
#         df.to_csv(f"{csv_path_prefix}_{key}.csv")

# use the function
read_npz_file(r"C:\Users\SDCNLab_P720\PycharmProjects\qsim\mlruns\0\8a7b061ceba4499f96dbd770c5a71e3e\artifacts\episodes_train\0\ep-00004_000001-0-r205-0800.npz")

# use the function
# npz_to_csv(
#     r"C:\Users\SDCNLab_P720\PycharmProjects\qsim\mlruns\0\8a7b061ceba4499f96dbd770c5a71e3e\artifacts\episodes_train\0\ep-00004_000001-0-r205-0800.npz",
#     r"C:\Users\SDCNLab_P720\Desktop\Yida\npz"
# )
