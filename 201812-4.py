def input_integers():
    return list(map(lambda x: int(x), input().split()))


def find(x):
    if x == fa[x]:
        return x
    fa[x] = find(fa[x])
    return fa[x]


n = int(input())
m = int(input())
r = int(input())

edges = []
fa = []

for i in range(n):
    fa.append(i)

for i in range(m):
    v, u, t = input_integers()
    v -= 1
    u -= 1
    edges.append((v, u, t))

edges.sort(key=lambda x: x[-1])

ans = 0
for e in edges:
    u, v, t = e
    if find(u) != find(v):
        fa[find(u)] = find(v)
        ans = t

print(ans)
