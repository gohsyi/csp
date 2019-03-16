def input_integers():
    return list(map(lambda x: int(x), input().split()))


n = int(input())

for _ in range(n):
    sta = []
    for r in range(3):
        sta.append(input_integers())
    Q, head, tail = [sta], 0, 1

    while head < tail:
        sta = Q[head]
        head += 1

