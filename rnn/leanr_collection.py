# !/usr/bin/python
# coding=utf-8
dict = {'world': 100, 'hello': 200, 'the': 10000}
count_pairs = sorted(dict.items(), key= lambda x: -x[1])
print(count_pairs)
zz, yy = zip(*count_pairs)
print(list(zz))

a = [1, 2, 3]
b = [2,3,4]
d = 'abcde'
zz = zip(a, b, d)
print(zz)


x, y, z = zip(*zz)
print(x)
print(y)
print(z)
