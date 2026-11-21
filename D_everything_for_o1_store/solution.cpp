#include <cstdio>
#include <cstdlib>
#include <vector>
// Line L = Q1 (front..back) followed by reversed Q2 (back..front).
// Closing a counter leaves L unchanged and only moves the boundary to an end of L;
// opening a counter moves the boundary to the middle. So: a doubly linked list with
// pops at both ends, insertion at the boundary, and a continuously maintained middle pointer.
int main() {
    size_t cap = 1 << 16, len = 0;
    char *buf = (char *)malloc(cap);
    for (;;) {
        size_t r = fread(buf + len, 1, cap - len, stdin);
        len += r;
        if (len < cap) break;
        cap *= 2; buf = (char *)realloc(buf, cap);
    }
    size_t pos = 0; long long n = 0;
    while (pos < len && (buf[pos] < '0' || buf[pos] > '9')) pos++;
    while (pos < len && buf[pos] >= '0' && buf[pos] <= '9') n = n * 10 + (buf[pos++] - '0');
    while (pos < len && (buf[pos] == ' ' || buf[pos] == '\n' || buf[pos] == '\r' || buf[pos] == '\t')) pos++;

    const int HEAD = 0, TAIL = 1;
    std::vector<int> nx(n + 3), pv(n + 3);
    nx[HEAD] = TAIL; pv[TAIL] = HEAD;
    std::vector<char> out; out.reserve(n + 2);
    long long m = 0, s1 = 0, mi = 0;      // total size, |Q1| (= index of bnd), index of mid
    int mid = TAIL, bnd = TAIL, cnt = 0;  // mid = node at index m/2, bnd = first node of the right part
    for (long long i = 0; i < n && pos < len; i++, pos++) {
        char c = buf[pos];
        bool changed = false;
        switch (c) {
            case 'a': case 'b': {
                int x = ++cnt + 1, p = pv[bnd];
                nx[p] = x; pv[x] = p; nx[x] = bnd; pv[bnd] = x;
                if (s1 <= mi) mi++;
                if (c == 'a') s1++; else bnd = x;
                m++; changed = true; break;
            }
            case 'A': {
                int x = nx[HEAD];
                out.push_back(char('0' + (x - 1) % 10));
                if (mid == x) mid = nx[x]; else mi--;
                nx[HEAD] = nx[x]; pv[nx[x]] = HEAD;
                s1--; m--; changed = true; break;
            }
            case 'B': {
                int x = pv[TAIL];
                out.push_back(char('0' + (x - 1) % 10));
                if (bnd == x) bnd = TAIL;
                if (mid == x) { mid = TAIL; mi = m - 1; }
                pv[TAIL] = pv[x]; nx[pv[x]] = TAIL;
                m--; changed = true; break;
            }
            case '>': s1 = 0; bnd = nx[HEAD]; break;
            case ']': s1 = m; bnd = TAIL; break;
            case '<': s1 = m / 2; bnd = mid; break;
            case '[': s1 = m - m / 2; bnd = (m & 1) ? nx[mid] : mid; break;
            default: i--; break;          // stray whitespace
        }
        if (changed) {
            long long t = m / 2;
            while (mi < t) { mid = nx[mid]; mi++; }
            while (mi > t) { mid = pv[mid]; mi--; }
        }
    }
    out.push_back('\n');
    fwrite(out.data(), 1, out.size(), stdout);
    return 0;
}
