def ones(n):
    return (1 << n) - 1


def includes(a, b):
    return a[1] <= b[1] and (a[0] >> (32 - a[1])) == (b[0] >> (32 - a[1]))


def legal(ip):
    return (ip[0] & ones(32-ip[1])) == 0


def format(ip):
    return '%i.%i.%i.%i/%i' % (
        (ip[0] >> 24) & ones(8),
        (ip[0] >> 16) & ones(8),
        (ip[0] >> 8) & ones(8),
        ip[0] & ones(8),
        ip[1])


def merge_equal(a, b, a_):
    alow, ahigh = a[0], a[0] + (1 << (32-a[1]))
    blow, bhigh = b[0], b[0] + (1 << (32-b[1]))
    a_low, a_high = a_[0], a_[0] + (1 << (32-a_[1]))
    if alow > bhigh or ahigh < blow:
        return False
    return min(alow, blow) == a_low and max(ahigh, bhigh) == a_high


def main():
    n = int(input())

    ip_list = []

    for i in range(n):
        ip = [0, 0, 0, 0]
        raw = input().split('/')
        if len(raw) > 1:
            length = int(raw[1])
        else:
            length = 8 * len(raw[0].split('.'))
        for i, num in enumerate(raw[0].split('.')):
            ip[i] = int(num)
        ip = ip[0]*(1<<24) + ip[1]*(1<<16) + ip[2]*(1<<8) + ip[3]
        ip_list.append((ip, length))

    ip_list.sort(key=lambda x: (x[0], x[1]))

    ip_list2 = [ip_list[0]]
    for i in range(1, len(ip_list)):
        a = ip_list2[-1]
        b = ip_list[i]
        if not includes(a, b):
            ip_list2.append(b)

    prev, next = [], []
    for i in range(len(ip_list2)):
        prev.append(i - 1)
        next.append(i + 1)
    next[-1] = -1

    for i in range(1, len(ip_list2)):
        ia = prev[prev[i]] if prev[prev[i]] > 0 else prev[i]
        a = ip_list2[ia]
        b = ip_list2[i]
        a_ = (a[0], a[1]-1)
        if a[1] == b[1] and legal(a_) and merge_equal(a, b, a_):
            ip_list2[ia] = a_
            next[ia] = next[i]
            prev[i + 1] = ia

    i = 0
    while i >= 0:
        print(format(ip_list2[i]))
        i = next[i]


main()
