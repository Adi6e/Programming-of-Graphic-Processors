from random import randint

n = int(input())
print(n)
first = list()
second = list()
for _ in range(n):
    first.append(randint(-1000, 1000))
for _ in range(n):
    second.append(randint(-1000, 1000))
for elem in first:
    print(elem, end=' ')
print('')
for elem in second:
    print(elem, end=' ')