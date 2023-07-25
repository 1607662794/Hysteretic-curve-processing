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

# 示例数据
force = [1, -2, 3, -4, 3, 2, 1, -2, 3, -4, 3, 2, 0, 1, 2]
zero_number = [0, 5, 10, 12, 14]

# 求得每一段上force绝对值最大的点的force序号
max_abs_force_indices = find_max_abs_force_indices(force, zero_number)
print("每一段上force绝对值最大的点的force序号：", max_abs_force_indices)
