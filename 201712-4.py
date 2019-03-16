def input_integers():
    return list(map(lambda x: int(x), input().split()))


n, m = input_integers()

for _ in range(m):
    t, a, b, c = input_integers()
