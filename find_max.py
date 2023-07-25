import numpy as np


def find_max_abs_force_indices(force, zero_number):
    max_abs_force_indices = []

    for i in range(len(zero_number) - 1):
        start_idx = zero_number[i]
        end_idx = zero_number[i + 1]
        segment_force = force[start_idx:end_idx]

        abs_force_values = [abs(val) for val in segment_force]
        max_abs_force_idx = start_idx + abs_force_values.index(max(abs_force_values))
        max_abs_force_indices.append(max_abs_force_idx)

    return max_abs_force_indices

if __name__ == "__main__":
    Input_dir = r"E:\Code\Hysteretic curve processing\sampling_data\RS3.csv"  # 原数据
    data = np.genfromtxt(Input_dir, delimiter=',', skip_header=1,
                         dtype=[('image_names', 'U50'), ('u [mm]', float), ('Fh [kN]', float)])
    force = data['Fh_kN']
    # 示例数据
    zero_number = [0, 29, 69, 104, 140, 191, 249, 304, 359, 443, 541, 637, 712, 774, 838, 905, 968, 1051, 1125, 1214, 1277, 1345, 1405, 1476, 1530, 1591, 1645, 1707, 1756, 1811, 1868, 1931, 1987, 2044, 2107, 2165, 2223, 2280, 2343, 2400, 2461, 2523, 2593]

    # 求得由零点划分的每一段上力值的绝对值最大的点的滞回曲线序号
    result = find_max_abs_force_indices(force.tolist(), zero_number)
    print("力值绝对值最大的点的滞回曲线序号：", result)
    print("力值绝对值最大的点数量：", len(result))
