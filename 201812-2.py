def input_integers():
    return list(map(lambda x: int(x), input().split()))


r, y, g = input_integers()
n = int(input())
total_time = 0

for i in range(n):
    k, t = input_integers()
    if k == 0:
        total_time += t
    elif k == 2:
        total_time += max(0, y + r - (y - t + total_time) % (r + y + g))
    elif k == 1:
        total_time += max(0, y + r - (y + r - t + total_time) % (r + y + g))
    else:
        total_time += max(0, y + r - (y + r + g - t + total_time) % (r + y + g))

print(total_time)
