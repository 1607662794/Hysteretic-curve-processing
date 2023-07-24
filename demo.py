import math

def dot_product(vector1, vector2):
    return sum(a * b for a, b in zip(vector1, vector2))

def vector_length(vector):
    return math.sqrt(sum(a * a for a in vector))

def vector_angle(vector1, vector2):
    dot_product_value = dot_product(vector1, vector2)
    vector1_length = vector_length(vector1)
    vector2_length = vector_length(vector2)

    if vector1_length == 0 or vector2_length == 0:
        raise ValueError("向量长度不能为0")

    cos_theta = dot_product_value / (vector1_length * vector2_length)
    theta = math.acos(cos_theta)
    theta_degrees = math.degrees(theta)
    return theta_degrees

if __name__ == "__main__":
    # 示例向量
    vector1 = [1, 2, 3]
    vector2 = [4, 5, 6]

    # 计算夹角
    angle = vector_angle(vector1, vector2)

    # 判断是否大于120°
    if angle > 120:
        print("夹角大于120°")
    else:
        print("夹角不大于120°")
