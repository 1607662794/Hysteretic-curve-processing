import math
'''该文件用于计算两个向量的角度，我用于判断翻转点'''
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
    vector1 = [9.837381539-9.801780884, 30.04-33.37]
    vector2 = [9.837381539-9.836943331, 30.04-30.06]
    vector3 = [9.801780884-9.837381539, 33.37-30.04]
    vector4 = [9.801780884-9.445995562, 33.37-33.13]
    vector3 = [9.801780884-9.837381539, 33.37-30.04]
    vector4 = [9.801780884-9.445995562, 33.37-33.13]

    # 计算夹角
    angle = vector_angle(vector1, vector2)
    print(angle)

    # 计算夹角
    angle = vector_angle(vector3, vector4)
    print(angle)
