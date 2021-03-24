# -*- coding: utf-8 -*-
import sys

sys.setrecursionlimit(1000000)  # 压缩大文件实时会出现超出递归深度，故修改限制


# 定义哈夫曼树的节点类
class node(object):

    def __init__(self, value=None, left=None, right=None, father=None):
        self.value = value
        self.left = left
        self.right = right
        self.father = father

    def build_father(self, right):
        n = node(value=self.value + right.value, left=self, right=right)
        self.father = right.father = n
        return n

    def encode(self):
        if self.father == None:
            return b''
        if self.father.left == self:
            return node.encode(self.father) + b'0'  # 左节点编号'0'
        else:
            return node.encode(self.father) + b'1'  # 右节点编号'1'


# 哈夫曼树构建
def build_tree(m):
    if len(m) == 1:
        return m
    sorts = sorted(m, key=lambda x: x.value, reverse=False)
    n = node.build_father(sorts[0], sorts[1])
    sorts.pop(0)
    sorts.pop(0)
    sorts.append(n)
    return build_tree(sorts)


def encode(echo):
    for x in node_dict.keys():
        ec_dict[x] = node.encode(node_dict[x])
        print(type(x))
        if echo:  # 输出编码表（用于调试）
            print(x)
            print(ec_dict[x])


def encodefile(string):
    print("Starting encode...")

    bytes_width = 1  # 每次读取的字节宽度
    i = 0

    count = len(string)
    nodes = []  # 结点列表，用于构建哈夫曼树
    buff = [b''] * int(count)  # 初始化
    # f.seek(0)

    # 计算字符频率,并将单个字符构建成单一节点
    while i < count:  # 频率导入字典
        buff[i] = string[0:bytes_width]
        string = string[bytes_width:]
        # buff[i] = f.read(bytes_width)
        if count_dict.get(buff[i], -1) == -1:
            count_dict[buff[i]] = 0
        count_dict[buff[i]] = count_dict[buff[i]] + 1
        i = i + 1
    print("Read OK")
    print(count_dict)  # 输出权值字典,可注释掉
    for x in count_dict.keys():
        node_dict[x] = node(count_dict[x])
        nodes.append(node_dict[x])

    tree = build_tree(nodes)  # 哈夫曼树构建
    encode(False)  # 构建编码表
    print("Encode OK")

    head = sorted(count_dict.items(), key=lambda x: x[1], reverse=True)  # 对所有根节点进行排序
    bit_width = 1
    print("head:", head[0][1])  # 动态调整编码表的字节长度，优化文件头大小
    if head[0][1] > 255:
        bit_width = 2
        if head[0][1] > 65535:
            bit_width = 3
            if head[0][1] > 16777215:
                bit_width = 4
    print("bit_width:", bit_width)
    i = 0
    raw = 0b1  # 二进制的1
    last = 0
    string2 = b''

    string2 = string2 + int.to_bytes(len(ec_dict), 2, byteorder='big')   # 写出结点数量
    string2 = string2 + int.to_bytes(bit_width, 1, byteorder='big')  # 写出编码表字节宽度
    for x in ec_dict.keys():  # 编码文件头
        string2 = string2 + bytes(x, encoding='utf-8')
        string2 = string2 + int.to_bytes(count_dict[x], bit_width, byteorder='big')

    print('head OK')
    while i < count:  # 开始压缩数据
        for x in ec_dict[buff[i]]:
            raw = raw << 1
            if x == 49:  # 表示1
                raw = raw | 1  # 0000 0011
            if raw.bit_length() == 9:
                raw = raw & (~(1 << 8))  # 1111 11111
                string2 = string2 + int.to_bytes(raw, 1, byteorder='big')
                # o.write(int.to_bytes(raw, 1, byteorder='big'))
                # o.flush()
                raw = 0b1
                tem = int(i / len(buff) * 100)
                if tem > last:
                    print("encode:", tem, '%')  # 输出压缩进度
                    last = tem
        i = i + 1

    if raw.bit_length() > 1:  # 处理文件尾部不足一个字节的数据
        raw = raw << (8 - (raw.bit_length() - 1))
        raw = raw & (~(1 << raw.bit_length() - 1))
        string2 = string2 + int.to_bytes(raw, 1, byteorder='big')
    # print("File encode successful.")
    return string2


def decodefile(string2):
    print("Starting decode...")

    raw = 0
    last = 0

    eof = len(string2)
    string3 = b''
    count = int.from_bytes(string2[0:2], byteorder='big')  # 取出结点数量
    string2 = string2[2:]
    bit_width = int.from_bytes(string2[0:1], byteorder='big')  # 取出编码表字宽
    string2 = string2[1:]
    i = 0
    n = 3
    de_dict = {}
    while i < count:  # 解析文件头
        key = string2[0:1]
        string2 = string2[1:]
        # key = f.read(1)
        value = string2[0]
        n = n + 2
        string2 = string2[bit_width:]
        de_dict[key] = value
        i = i + 1
    for x in de_dict.keys():
        node_dict[x] = node(de_dict[x])
        nodes.append(node_dict[x])
    tree = build_tree(nodes)  # 重建哈夫曼树
    encode(False)  # 建立编码表
    for x in ec_dict.keys():  # 反向字典构建
        inverse_dict[ec_dict[x]] = x
    i = n
    data = b''
    while i < eof:  # 开始解压数据
        raw = int.from_bytes(string2[0:1], byteorder='big')
        string2 = string2[1:]
        i = i + 1
        j = 8
        while j > 0:
            if (raw >> (j - 1)) & 1 == 1:
                data = data + b'1'
                raw = raw & (~(1 << (j - 1)))
            else:
                data = data + b'0'
                raw = raw & (~(1 << (j - 1)))
            if inverse_dict.get(data, 0) != 0:
                string3 = string3 + inverse_dict[data]
                data = b''
            j = j - 1
        tem = int(i / eof * 100)
        if tem > last:
            print("decode:", tem, '%')  # 输出解压进度
            last = tem
        raw = 0
    print(string3)


if __name__ == '__main__':
    # 数据初始化
    node_dict = {}  # 建立原始数据与编码节点的映射，便于稍后输出数据的编码
    count_dict = {}  #  记录'0''1'数
    ec_dict = {}    # 编码字典
    nodes = []       #  结点列表
    inverse_dict = {}   # 反向字典
    origin_string = "111111111111111111111111111111111111111010101000000000000000000001100010"
    encode_string = encodefile(origin_string)
    print(encode_string)
    decodefile(encode_string)
