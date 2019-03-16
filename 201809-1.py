def input_integers():
    return list(map(lambda x: int(x), input().split()))


n = int(input())
price = input_integers()
res = []

for i in range(n):
    p = price[i]
    m = 1
    if i > 0:
        p += price[i - 1]
        m += 1
    if i < n-1:
        p += price[i + 1]
        m += 1

    print(int(p / m), end=' ')

