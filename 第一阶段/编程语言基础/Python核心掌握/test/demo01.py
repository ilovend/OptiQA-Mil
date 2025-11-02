# import keyword
# print("Hello, World!")


# def is_valid_number(name):
#     try:
#         exec(f"{name} = None")  # 尝试将 None 分配给变量名
#         return True
#     except:
#         return False
# print(is_valid_number("var1"))  # True
# print(is_valid_number("1var"))  # False

# print(keyword.kwlist)  # True

# if False:
#     print("true")
# else:
#     print("false")

# item_one = 1
# item_two = 2
# item_three = 3
# total = item_one + \
#         item_two + \
#         item_three
# print(total)

# str="123456789"

# print(str[0:-1])
# print(str[1:5:2])

# a=123
# print(type(a))

# a=3+4j
# print(type(a))


# import numpy as np

# impedance = 50 + 30j  # 50Ω电阻 + 30Ω感抗
# current = 2 + 0j      # 2A电流
# voltage = impedance * current  # (100+60j)伏特
# print(f"电压: {voltage} V")  # 输出电压 

str=" Hello, World! "
print(str.strip())  # 去除首尾空格
print(str.lower())  # 转为小写
print(str.upper())  # 转为大写
print(str.replace("World", "Python"))  # 替换子字符串
print(str.split(","))  # 按逗号分割字符串
print(str.find("World"))  # 查找子字符串位置
print(len(str))  # 字符串长度
print(str.startswith(" Hello"))  # 检查是否以指定子字符串开头
print(str.endswith("! "))  # 检查是否以指定子字符串结尾
print(str.count("o"))  # 计算子字符串出现次数
print(str.index("World"))  # 获取子字符串索引位置
print(str.isalpha())  # 检查是否只包含字母
print(str.isdigit())  # 检查是否只包含数字
print(str.isspace())  # 检查是否只包含空白字符
print(str.capitalize())  # 首字母大写
print(str.title())  # 每个单词首字母大写
print(str.center(30, '*'))  # 居中并填充字符
print(str.zfill(20))  # 左侧填充零至指定长度
print(str.encode('utf-8'))  # 编码为字节
print(str.format())  # 格式化字符串
print(str.partition("World"))  # 分割字符串为三部分
print(str.rfind("o"))  # 反向查找子字符串位置
print(str.rsplit(" ", 1))  # 从右侧分割字符串
print(str.swapcase())  # 大小写互换
# print(str.translate(str.maketrans("Helo", "12345")))  # 字符替换
print(str.zfill(25))  # 左侧填充零至指定长度
print(str.expandtabs(4))  # 将制表符转换为空格
print(str.islower())  # 检查是否全为小写
print(str.isupper())  # 检查是否全为大写
print(str.lstrip())  # 去除左侧空格
print(str.rstrip())  # 去除右侧空格
