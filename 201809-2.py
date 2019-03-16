def input_integers():
    return list(map(lambda x: int(x), input().split()))


n = int(input())
timeline = []
ans = 0

for _ in range(1000001):
    timeline.append(0)

for _ in range(n + n):
    a, b = input_integers()
    for i in range(a, b):
        if timeline[i] > 0:
            ans += 1
        else:
            timeline[i] = 1

print(ans)
