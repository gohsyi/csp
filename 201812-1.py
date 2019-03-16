def input_int():
    return list(map(lambda x: int(x), input().split()))

r, g, b = input_int()
n = int(input())
ans = 0

for i in range(n):
    k, t = input_int()
    if k == 0:
        ans += t
    elif k == 1:  # red
        ans += t
    elif k == 2:
        ans += t + r

print(ans)
