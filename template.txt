#define M (L+R>>1)
 
 
// 删除标记线段树
struct SegmentTree {
	int y1, y2;
	int sumv[maxn<<2], addv[maxn<<2];
	bool clr[maxn<<2];
	
	void init() {
		sumv[1] = addv[1] = 0;
		clr[1] = 1; // 删除整棵树
	}
		
	#define lc (o<<1)
	#define rc (o<<1^1)
	
	void update(int o) {
		sumv[o] = sumv[lc] + sumv[rc];
	}
	
	void pushdown(int o, int L, int R) {
		if(clr[o]) {
			sumv[lc] = sumv[rc] = addv[lc] = addv[rc] = 0;
			clr[lc] = clr[rc] = 1;
			clr[o] = 0;
		}
		if(addv[o]) {
			int& x = addv[o];
			addv[lc] += x;
			addv[rc] += x;
			sumv[lc] += (M-L+1) * x; // 让子结点的标记立刻生效
			sumv[rc] += (R-M) * x;
			x = 0;
		}
	}
	
	void modify(int o, int L, int R) {
		if(y1 <= L && R <= y2) {
			addv[o]++;
			sumv[o] += (R-L+1); // 标记立刻生效
		} else {
			pushdown(o, L, R);
			if(y1 <= M) modify(lc, L, M);
			if(y2 > M) modify(rc, M+1, R);
			update(o);
		}
	}
	
	int query(int o, int L, int R) {
		if(y1 <= L && R <= y2) return sumv[o];
		pushdown(o, L, R);
		int ret = 0;
		if(y1 <= M) ret += query(lc, L, M);
		if(y2 > M) ret += query(rc, M+1, R);
		return ret;
	}
	
	#undef lc
	#undef rc
};
 
 
// 动态开点线段树
struct DynamicSegmentTree {
	int n;
	int d, x, y1, y2;
	int maxv[maxn<<2], sumv[maxn<<2];
	
	#define lc lc[o]
	#define rc rc[o]
	
	void update(int o) {
		sumv[o] = sumv[lc] + sumv[rc];
		maxv[o] = max(maxv[lc], maxv[rc]);
	}
	
	void modify(int& o, int L, int R) {
		if(!o) o = ++n;
		if(L == R) maxv[o] = sumv[o] = d;
		else {
			if(x <= M) modify(lc, L, M);
			else modify(rc, M+1, R);
			update(o); // notice
		}
	}
	
	int query_sum(int o, int L, int R) {
		if(!o) return 0;
		if(y1 <= L && R <= y2) return sumv[o];
		int ret = 0;
		if(y1 <= M) ret += query_sum(lc, L, M);
		if(M < y2) ret += query_sum(rc, M + 1, R;
		return ret;
	}
	
	int query_max(int o, int L, int R, int y1, int y2) {
		if(!o) return 0;
		if(y1 <= L && R <= y2) return maxv[o];
		int ret = 0;
		if(y1 <= M) ret = max(ret, query_max(lc, L, M));
		if(M < y2) ret = max(ret, query_max(rc, M + 1, R));
		return ret;
	}
	
	#undef lc
	#undef rc
};
 
 
// 主席树
struct FunctionalSegmentTree {
	int n;
	int d, k;
	int lc[maxn], rc[maxn], s[maxn];
	
	void build(int& x, int y, int L, int R) {
		x = ++n;
		lc[x] = lc[y]; rc[x] = rc[y];
		if(L == R) v[x] = v[y] + 1;
		else if(d <= T[M]) build(lc[x], lc[y], L, M);
			else build(rc[x], rc[y], M+1, R);
		s[x] = v[x] + s[lc[x]] + s[rc[x]]; // OR : s[x] = s[y] + 1
	}
	
	int query(int x, int y, int L, int R) {
		if(L == R) return T[L];
		int ls = s[lc[y]] - s[lc[x]];
		if(ls >= k) return query(lc[x], lc[y], L, M);
		k -= ls;
		return query(rc[x], rc[y], M+1, R);
	}
};
 
 
struct BIT {
	int n, c[maxn];
	void modify(int x, int d) {
		for(; x <= n; x += (x&-x)) c[x] += d;
	}
	int query(int x) {
		int ret = 0;
		for(; x; x -= (x&-x)) ret += c[x];
		return ret;
	}
	void clear(int x) {
		for(; x <= n; x += (x&-x)) c[x] = 0;
	}
};
 
 
struct UnionFindSet {
	int n, pa[maxn];
	
	void init() {
		for(int i = 1; i <= n; i++) pa[i] = i;
	}
	
	int find(int x) {
		return x == pa[x] ? x : (pa[x] = find(pa[x]));
	}
};
 
 
struct HASH {
	int n;
	int A[maxn], head[maxn], next[maxn], hash[maxn];
	
	void init() {
		n = 0;
		memset(A, 0, sizeof(A));
		memset(head, 0, sizeof(head));
		memset(next, 0, sizeof(next));
		memset(hash, 0, sizeof(hash));
	}
	
	bool find(int x) {
		int y = (x >= 0 ? x : -x) % maxn;
		if(!hash[y]) return 0;
		for(int i = head[y]; i; i = next[i]) if(A[i] == x) return 1;
		return 0;
	}
	
	void insert(int x) {
		int y = (x >= 0 ? x : -x) % maxn;
		hash[y] = 1; A[++n] = x;
		next[n] = head[y]; head[y] = n;
	}
};
 
 
struct TreeChain {
	int n;
	int s[maxn], tid[maxn], dep[maxn], son[maxn], top[maxn];
	
	void init() {
		dfs1(1, 1);
		dfs2(1, 1);
	}
	
	void dfs1(int x, int y) {
		s[x] = 1; dep[x] = dep[fa[x] = y] + 1;
		for(int i = 0; i < ch[x].size(); i++) {
			int z = ch[x][i];
			if(z == fa[x]) continue;
			dfs1(z, x);
			s[x] += s[z];
			if(s[z] > s[son[x]]) son[x] = z;
		}
	}
 
	void dfs2(int x, int y) {
		tid[x] = ++n; top[x] = y;
		if(son[x]) dfs2(son[x], y);
		for(int i = 0; i < ch[x].size(); i++) {
			int z = ch[x][i];
			if(z == fa[x] || z == son[x]) continue;
			dfs2(z, z);
		}
	}
	
	int query_sum(int x, int y) {
		int ret = 0, cx = c[x]; // 用多棵动态开点线段树维护
		while(top[x] != top[y]) {
			if(dep[top[x]] < dep[top[y]]) swap(x, y);
			seg.y1 = tid[top[x]];
			seg.y2 = tid[x];
			ret += seg.query_sum(root[cx], 1, n);
			x = fa[top[x]];
		}
		if(dep[x] > dep[y]) swap(x, y);
		seg.y1 = tid[x];
		seg.y2 = tid[y];
		ret += query_sum(root[cx], 1, n);
		return ret;
	}
	
	int query_max(int x, int y) {
		int ret = 0, cx = c[x];
		while(top[x] != top[y]) {
			if(dep[top[x]] < dep[top[y]]) swap(x, y);
			seg.y1 = tid[top[x]];
			seg.y2 = tid[x];
			ret = max(ret, query_max(root[cx], 1, n)); // notice
			x = fa[top[x]];
		}
		if(dep[x] > dep[y]) swap(x, y);
		seg.y1 = tid[x];
		seg.y2 = tid[y];
		ret = max(ret, query_max(roots[cx], 1, n));
		return ret;
	}
};
 
 
struct Treap {
	int n;
	int s[maxn], r[maxn], ch[maxn][2];
	
	#define lc ch[o][0]
	#define rc ch[o][1]
	
	void update(int o) {
		s[o] = s[lc] + s[rc] + 1;
	}
	
	void rotate(int& o, int d) {
		int p = ch[o][d^1]; ch[o][d^1] = ch[p][d]; ch[p][d] = o;
		update(o); update(p); o = p;
	}
	
	void insert(int& o, int k) {
		if(!o) r[o = ++n] = rand(), s[o] = 1;
		else {
			s[o]++;
			int d = (k <= s[lc] ? 0 : 1);
			if(d == 1) k -= (s[lc] + 1);
			insert(ch[o][d], k);
			if(r[ch[o][d]] > r[o]) rotate(o, d^1);
		}
	}
	
	#undef lc
	#undef rc
};
 
 
// 维护数列
struct Splay {
	int ch[maxn][2];
	int s[maxn];
	bool rev[maxn];
	
	int n, m;
	int ch[maxn][2], id[maxn];
	int mls[maxn], mrs[maxn], maxs[maxn], s[maxn], rech[maxn], v[maxn], sumv[maxn];
	bool mdf[maxn], rev[maxn];
	
	#define lc ch[o][0]
	#define rc ch[o][1]
	
	void update(int o) {
		s[o] = s[lc] + s[rc] + 1;
		sumv[o] = sumv[lc] + sumv[rc] + v[o];
		maxs[o] = max(maxs[lc], maxs[rc]);
		maxs[o] = max(maxs[o], mrs[lc] + v[o] + mls[rc]);
		mls[o] = max(mls[lc], sumv[lc] + v[o] + mls[rc]);
		mrs[o] = max(mrs[rc], sumv[rc] + v[o] + mrs[lc]);
	}
	
	void pushdown(int o) {
		if(mdf[o]) {
			mdf[o] = rev[o] = 0;
			if(lc) mdf[lc] = 1, v[lc] = v[o], sumv[lc] = v[o]*s[lc];
			if(rc) mdf[rc] = 1, v[rc] = v[o], sumv[rc] = v[o]*s[rc];
			if(v[o] >= 0) {
				if(lc) mls[lc] = mrs[lc] = maxs[lc] = sumv[lc];
				if(rc) mls[rc] = mrs[rc] = maxs[rc] = sumv[rc];
			} else {
				if(lc) mls[lc] = mrs[lc] = 0, maxs[lc] = v[o];
				if(rc) mls[rc] = mrs[rc] = 0, maxs[rc] = v[o];
			}
		}
		if(rev[o]) {
			rev[o]^=1;
			rev[lc]^=1;
			rev[rc]^=1;
			swap(mls[lc], mrs[lc]);
			swap(mls[rc], mrs[rc]);
			swap(ch[lc][0], ch[lc][1]);
			swap(ch[rc][0], ch[rc][1]);
		}
	}
	
	void rotate(int& o, int d) {
		int p = ch[o][d^1]; ch[o][d^1] = ch[p][d]; ch[p][d] = o;
		update(o); update(p); o = p;
	}
	
	int cmp(int o, int k) {
		if(s[lc] + 1 == k) return -1;
		return k < s[lc] + 1 ? 0 : 1;
	}
	
	void splay(int& o, int k) {
		pushdown(o);
		int d = cmp(o, k);
		if(d == -1) return;
		if(d == 1) k -= (s[lc] + 1);
		int p = ch[o][d];
		pushdown(p);
		int d2 = cmp(p, k);
		int k2 = (d2 == 0 ? k : k-s[ch[p][0]]-1);
		if(d2 != -1) {
			splay(ch[p][d2], k2);
			if(d == d2) rotate(o, d^1); else rotate(ch[o][d], d);
		}
		rotate(o, d^1);
	}
	
	void build(int& o, int L, int R) {
		if(L > R) return;
		o = (L+R)>>1;
		build(lc, L, o-1);
		build(rc, o+1, R);
		update(o);
	}
	
	void getInterval(int& o, int pos, int tot) {
		splay(o, pos);
		splay(rc, pos + tot - s[lc]);
	}
	
	// 用的时候改一下 
	int o, pos, tot, val;
	
	void insert() {
		getInterval(o, pos+1, 0);
		for(int i = n+1; i <= n+tot; i++) scanf("%d", &v[i]);
		build(ch[rc][0], n+1, n+tot);
		n += tot; //
		update(rc); update(o);
	}
	
	void remove() {
		getInterval(o, pos, tot);
		ch[rc][0] = 0;
		update(rc); update(o);
	}
	
	void modify() {
		getInterval(o, pos, tot);
		int p = ch[rc][0];
		v[p] = val;
		mdf[p] = 1;
		sumv[p] = v[p]*s[p];
		if(val >= 0) mls[p] = mrs[p] = maxs[p] = sumv[p];
		else mls[p] = mrs[p] = 0, maxs[p] = v[p]; // 规定 mls >= 0, mrs >= 0, 方便维护
		update(rc); update(o);
	}
	
	void rever() {
		getInterval(o, pos, tot);
		if(!mdf[o]) {
			int p = ch[rc][0];
			rev[p]^=1;
			swap(ch[p][0], ch[p][1]);
			swap(mls[p], mrs[p]);
		}
		update(rc); update(o);
	}
	
	void query() {
		getInterval(o, pos, tot);
		printf("%d\n", sumv[ch[rc][0]]);
	}
	
	#undef lc
	#undef rc
};
 
 
struct Splay {
	int s[maxn], ch[maxn][2];
	
	#define lc ch[o][0]
	#define rc ch[o][1]
	
	int kth(int o, int k) {
		if(s[lc] + 1 == k) return o;
		else if(s[lc] >= k) return kth(lc, k); 
		else return kth(rc, k-s[lc]-1); 
	}
	
	// 自底向上
	void rotate(int& o, int x)  {
		int y = fa[x], z = fa[y];
		bool d = (ch[y][0] == x), d2 = (ch[z][0] == y);
		if(o == y) o = x; else ch[z][d2] = x;
		fa[x] = z; fa[y] = x; fa[ch[y][d] = ch[x][d^1]] = y;
		ch[x][d^1] = y; update(y); update(x);
	}
	
	void splay(int& o, int x) {
		while(o != x) {
			int y = fa[x], z = fa[y]; 
			if(o != y)
				if(ch[y][0] == x^ch[z][0] == y) rotate(o, x); else rotate(o, y);
			rotate(o, x);
		}
	}
	
	void remove(int& o, int k) {
		int p = kth(o, k-1), q = kth(o, k+1); 
		splay(o, p); splay(rc, q);
		int t = ch[q][0]; ch[q][0] = 0; fa[t] = s[t] = 0;
		update(q); update(p);
	}
	
	#undef lc
	#undef rc
};
 
 
// 线段树的功能
struct Splay {
	int s[maxn], ch[maxn][2];
	lli v[maxn], sumv[maxn], inc[maxn];
	
	#define lc ch[o][0]
	#define rc ch[o][1]
	#define seq ch[rc][0]
	
	void update(int o) {
		s[o] = s[lc] + s[rc] + 1;
		sumv[o] = sumv[lc] + sumv[rc] + v[o];
	}
	
	void pushdown(int o) {
		if(!inc[o]) return;
		inc[lc] += inc[o]; v[lc] += inc[o]; sumv[lc] += inc[o] * s[lc];
		inc[rc] += inc[o]; v[rc] += inc[o]; sumv[rc] += inc[o] * s[rc];
		inc[o] = 0;
	}
	
	int cmp(int o, int k) {
		if(s[lc] + 1 == k) return -1;
		return k < s[lc] + 1 ? 0 : 1;
	}
	
	void rotate(int& o, int d) {
		int p = ch[o][d^1]; ch[o][d^1] = ch[p][d]; ch[p][d] = o;
		update(o); update(p); o = p;
	}
	
	void splay(int& o, int k) {
		pushdown(o); //
		int d = cmp(o, k);
		if(d == -1) return;
		if(d == 1) k -= (s[lc] + 1);
		int p = ch[o][d];
		pushdown(p); //
		int d2 = cmp(p, k);
		if(d2 != -1) {
			int k2 = (d2 == 0 ? k : k-s[ch[p][0]]-1);
			splay(ch[p][d2], k2);
			if(d == d2) rotate(o, d^1); else rotate(ch[o][d], d);
		}
		rotate(o, d^1);
	}
	
	void build(int& o, int L, int R) {
		if(L > R) return;
		o = (L+R)>>1;
		build(lc, L, o-1); build(rc, o+1, R);
		update(o);
	}
};
 
 
struct KMP {
	int n, m;
	int f[maxn];
	char P[maxn];
	
	void getFail() {
		f[0] = f[1] = 0;
		for(int i = 1; i < m; i++) { // 用 f[i] 推出 f[i+1]
			int j = f[i];
			while(j && P[i] != P[j]) j = f[j];
			f[i+1] = (P[i] == P[j] ? j+1 : 0);
		}
	}
	
	void find(char* T) {
		n = strlen(T), m = strlen(P);
		getFail();
		int j = 0;
		for(int i = 0; i < n; i++) {
			while(j && P[j] != T[i]) j = f[j];
			if(P[j] == T[i]) j++;
			if(j == m) { printf("%d\n", i-m+2); break; }
		}
	}
};
 
 
struct SuffixArray {
	int n;
	int s[maxn], sa[maxn], rank[maxn], height[maxn]
	int t[maxn], t2[maxn], c[maxn];
 
	void build_sa(int m) {
		int i, *x = t, *y = t2; // 加快交换时间
		// 基数排序
		for(i = 0; i < m; i++) c[i] = 0;
		for(i = 0; i < n; i++) c[x[i] = s[i]]++;
		for(i = 1; i < m; i++) c[i] += c[i-1];
		for(i = n-1; i >= 0; i--) sa[--c[x[i]]] = i;
		for(int k = 1; k <= n; k <<= 1) {
			int p = 0;
			for(i = n-k; i < n; i++) y[p++] = i;
			for(i = 0; i < n; i++) if(sa[i] >= k) y[p++] = sa[i]-k; // 用上次结果对第二关键字进行排序
			for(i = 0; i < m; i++) c[i] = 0;
			for(i = 0; i < n; i++) c[x[y[i]]]++;
			for(i = 0; i < m; i++) c[i] += c[i-1];
			for(i = n-1; i >= 0; i--) sa[--c[x[y[i]]]] = y[i];
			swap(x, y);
			p = 1;
			x[sa[0]] = 0;
			for(i = 1; i < n; i++)
				x[sa[i]] = y[sa[i-1]]==y[sa[i]] && y[sa[i-1]+k]==y[sa[i]+k] ? p-1 : p++;
			if(p >= n) break;
			m = p;
		}
	}
 
	void build_height() {
		int i, j, k = 0;
		for(i = 0; i < n; i++) rank[sa[i]] = i;
		for(i = 0; i < n; i++) {
			if(k) k--;
			int j = sa[rank[i]-1];
			while(s[i+k] == s[j+k]) k++;
			height[rank[i]] = k;
		}
	}
};


struct SPFA {
	int n, m, s, t;
	int d[maxn];
	bool ban[maxn], inq[maxn];
	vector edges;
	vector G[maxn];
 
	void init(int n, int s, int t) {
		this->n = n;
		this->s = s;
		this->t = t;
	}
 
	void AddEdge(int from, int to, int dist) {
		edges.push_back((Edge) {from, to, dist});
		edges.push_back((Edge) {to, from, dist});
		m = edges.size();
		G[from].push_back(m-2);
		G[to].push_back(m-1);
	}
 
	int spfa(int x, int y) {
		queue Q;
		memset(d, 0x3f, sizeof(d));
		Q.push(s);
		inq[s] = 1;
		d[s] = 0;
 
		while(!Q.empty()) {
			int u = Q.front();
			Q.pop();
			inq[u] = 0;
			for(int i = 0; i < G[u].size(); i++) {
				Edge& e = edges[G[u][i]];
				if(!ban[e.to] && d[e.to] > d[u] + e.dist) {
					d[e.to] = d[u] + e.dist;
					if(!inq[e.to]) Q.push(e.to), inq[e.to] = 1;
				}
			}
		}
		return d[t];
	}
};
 
 
struct Dijkstra {
	int n, m;
	vector edges;
	vector G[maxn];
	bool done[maxn];
	int d[maxn], p[maxn];
 
	void init(int n) {
		this->n = n;
		for(int i = 0; i < n; i++) G[i].clear();
		edges.clear();
	}
 
	void AddEdge(int from, int to, int dist) {
		edges.push_back((Edge) {from, to, dist});
		m = edges.size();
		G[from].push_back(m-1);
	}
 
	void dijkstra(int s) {
		priority_queue Q; // HeapNode 自定义结构体
		                            // 存储二元组 {结点距s距离(dist), 结点编号(u)}
		                            // 并根据 dist 进行排序
		for(int i = 0; i < n; i++) d[i] = INF;
		d[s] = 0;
		memset(done, 0, sizeof(done)); // 永久编号
		Q.push((HeapNode){0, s});
		while(!Q.empty()) {
			HeapNode x = Q.top(); Q.pop();
			int u = x.u;
			if(done[u]) continue;
			done[u] = true;
			for(int i = 0; i < G[u].size(); i++) {
				Edge& e = edges[G[u][i]];
				if(d[e.to] > d[u] + e.dist) {
					d[e.to] = d[u] + e.dist;
					p[e.to] = G[u][i];
					Q.push((HeapNode){d[e.to], e.to});
				}
			}
		}
	}
 
	void GetShortestPaths(int s, int* dist, vector* paths) {
		dijkstra(s);
		for(int i = 0; i < n; i++) {
			dist[i] = d[i];
			paths[i].clear();
			int t = i;
			paths[i].push_back(t);
			while(t != s) {
				paths[i].push_back(edges[p[t]].from);
				t = edges[p[t]].from;
			}
			reverse(paths[i].begin(), paths[i].end());
		}
	}
};
 
 
struct MST {
	int n, m;
 
	void AddEdge(int from, int to, int dist) {
		edges.push_back((Edge){from, to, dist});
		m = edges.size();
		G[from].push_back(m-1);
	}
 
	int mst() {
		int ans = 0;
		sort(edges.begin(), edges.end());
		for(int i = 0; i < m; i++) {
			Edge& e = edges[i];
			int x = ufs.find(e.from), y = ufs.find(e.to);
			if(x != y) ufs.unin(x, y), ans += e.dist;
		}
		return ans;
	}
};
 
 
struct TwoSAT {
	int n;
	vector G[maxn*2];
	bool mark[maxn*2];
	int S[maxn*2], c;
 
	bool dfs(int x) {
		if (mark[x^1]) return false;
		if (mark[x]) return true;
		mark[x] = true;
		S[c++] = x;
		for (int i = 0; i < G[x].size(); i++)
			if (!dfs(G[x][i])) return false;
		return true;
	}
 
	void init(int n) {
		this->n = n;
		for (int i = 0; i < n*2; i++) G[i].clear();
		memset(mark, 0, sizeof(mark));
	}
 
	// x = xval or y = yval
	void add_clause(int x, int xval, int y, int yval) {
		x = x * 2 + xval;
		y = y * 2 + yval;
		G[x^1].push_back(y);
		G[y^1].push_back(x);
	}
 
	bool solve() {
		for(int i = 0; i < n*2; i += 2)
			if(!mark[i] && !mark[i+1]) {
				c = 0;
				if(!dfs(i)) {
					while(c > 0) mark[S[--c]] = false;
					if(!dfs(i+1)) return false;
				}
			}
		return true;
	}
};
 
 
// 判定二分图
bool bipartite(int u, int b) {
	for(int i = 0; i < G[u].size(); i++) {
		int v = G[u][i];
		if(bccno[v] != b) continue;
		if(color[v] == color[u]) return false;
		if(!color[v]) {
			color[v] = 3 - color[u];
			if(!bipartite(v, b)) return false;
		}
	}
	return true;
}
 
 
// 双联通分量算法
 
stack S;
 
int dfs(int u, int fa) {
	int lowu = pre[u] = ++dfs_clock;
	int child = 0;
	for(int i = 0; i < G[u].size(); i++) {
		int v = G[u][i];
		Edge e = (Edge){u, v};
		if(!pre[v]) { // 没有访问过v
			S.push(e);
			child++;
			int lowv = dfs(v, u);
			lowu = min(lowu, lowv); // 用后代的low函数更新自己
			if(lowv >= pre[u]) {
				iscut[u] = true;
				bcc_cnt++;
				bcc[bcc_cnt].clear();
				for(;;) {
					Edge x = S.top(); S.pop();
					if(bccno[x.u] != bcc_cnt) {
						bcc[bcc_cnt].push_back(x.u);
						bccno[x.u] = bcc_cnt;
					}
					if(bccno[x.v] != bcc_cnt) {
						bcc[bcc_cnt].push_back(x.v);
						bccno[x.v] = bcc_cnt;
					}
					if(x.u == u && x.v == v) break;
				}
			}
		} else if(pre[v] < pre[u] && v != fa) {
			S.push(e);
			lowu = min(lowu, pre[v]); // 用反向边更新自己
		}
	}
	if(fa < 0 && child == 1) iscut[u] = 0;
	return lowu;
}
 
void find_bcc(int n) {
	memset(pre, 0, sizeof(pre));
	memset(iscut, 0, sizeof(iscut));
	memset(bccno, 0, sizeof(bccno));
	dfs_clock = bcc_cnt = 0;
	for(int i = 0; i < n; i++)
		if(!pre[i]) dfs(i, -1);
}
 
 
struct Tarjan {
	vector G[maxn];
	int pre[maxn], lowlink[maxn], sccno[maxn], dfs_clock, scc_cnt;
	stack S;
 
	void dfs(int u) {
		pre[u] = lowlink[u] = ++dfs_clock;
		S.push(u);
		for(int i = 0; i < G[u].size(); i++) {
			int v = G[u][i];
			if(!pre[v]) {
				dfs(v);
				lowlink[u] = min(lowlink[u], lowlink[v]);
			} else if(!sccno[v]) { // notice
				lowlink[u] = min(lowlink[u], pre[v]);
			}
		}
		if(lowlink[u] == pre[u]) {
			scc_cnt++;
			for(;;) {
				int x = S.top(); S.pop();
				sccno[x] = scc_cnt;
				if(x == u) break;
			}
		}
	}
 
	void find_scc(int n) {
		dfs_clock = scc_cnt = 0;
		memset(sccno, 0, sizeof(sccno));
		memset(pre, 0, sizeof(pre));
		for(int i = 0; i < n; i++)
			if(!pre[i]) dfs(i);
	}
};
 
 
struct TOPO {
	int n, m, d[maxn], d0[maxn];
	vector edges;
	vector G[maxn];
 
	void init(int n) {
		this->n = n;
		memset(d0, 0, sizeof(d0));
		edges.clear();
		for(int u = 1; u <= n; u++) G[u].clear();
	}
 
	void AddEdge(int from, int to, int dist) {
		edges.push_back((Edge){to, dist});
		m = edges.size();
		G[from].push_back(m-1);
		d0[to]++;
	}
 
	int toposort() {
		queue Q;
		memset(d, 0, sizeof(d));
		for(int u = 1; u <= n; u++) if(G[u].size() != 0 && !d0[u]) Q.push(u);
		int max_d = 0;
 
		while(!Q.empty()) {
			int u = Q.front(); Q.pop();
			max_d = max(max_d, d[u]); // 最长路
			for(int i = 0; i < G[u].size(); i++) {
				Edge& e = edges[G[u][i]];
				d[e.to] = max(d[e.to], d[u] + e.dist);
				if(--d0[e.to] == 0) Q.push(e.to);
			}
		}
		return max_d;
	}
};
 
 
// 二分图最大基数匹配
struct Match {
	vector G[maxn];
	bool vis[maxn];
 
	int dfs(int u) {
		for(int i = 0; i < G[u].size(); i++) {
			int v = G[u][i];
			if(!vis[v]) {
				vis[v] = 1;
				if(y[v] == -1 || dfs(y[v])) {
					x[u] = v;
					y[v] = u;
					return 1;
				}
			}
		}
		return 0;
	}
 
	int match() {
		int ans = 0;
		memset(y, -1, sizeof(y));
		for(int i = 1; i <= m; i++) {
			memset(vis, 0, sizeof(vis));
			ans += dfs(i);
		}
		return ans;
	}
};
 
 
// 最小环
struct Floyed {
	int n;
	int G[maxn][maxn], f[maxn][maxn];
 
	int floyed() {
		int ans = INF;
		for(int i = 0; i < n; i++)
			for(int j = 0; j < n; j++)
				f[i][j] = G[i][j];
 
		// 可以直接枚举 k, i, j, 在计算 f 的值的同时更新 ans.
		// 即 :
		/*
		for(int k = 0; k < n; k++) {
		    for(int i = 0; i < n; i++) if(G[i][k] != INF)
		        for(int j = 0; j < n; j++) if(i != k && i != j && k != j) {
		            ans = min(ans, f[i][j] + G[i][k] + G[k][j]);
		            f[i][j] = min(f[i][j], f[i][k] + f[k][j]);
		        }
		}
		*/
		// 也是种可行的做法
 
		// 下面是优化过的做法 :
		for(int k = 0; k < n; k++) { // 枚举编号最大点
			for(int i = 0; i < k; i++) if(G[i][k] != INF)
					for(int j = i+1; j < k; j++) // 如果为有向边, j 从 0 开始枚举
						ans = min(ans, f[i][j] + G[i][k] + G[k][j]);
			// 此时 f[i][j] 一定没有用 k 更新过
			// 所以 f[i][j] 表示的路径不过 k
			// 更新最短路
			for(int i = 0; i < n; i++)
				for(int j = 0; j < n; j++)
					f[i][j] = min(f[i][j], f[i][k] + f[k][j]); // 用 f[][] 而不是 G[][] 在更新答案
		}
		return ans == INF ? -1 : ans;
	}
};
 
 
// 无向图欧拉路径
struct Euler {
	int n;
	bool G[maxn][maxn], vis[maxn][maxn];
	vector ans;
 
	void euler(int u) {
		for(int v = 0; v < n; v++) if(G[u][v] && !vis[u][v]) {
			vis[u][v] = vis[v][u] = 1;
			euler(v);
			ans.push_back((Edge){u, v});
		}
	}
 
	int judge(int start) {
		euler(start);
		if(ans.size() == n && ans[0].to == ans[ans.size()-1].from) return 1;
		return 0;
	}
};
 
 
struct LCA {
	int n;
	int fa[maxn];   // 父亲数组
	int cost[maxn]; // 和父亲的费用
	int L[maxn];    // 层次（根节点层次为0）
	int anc[maxn][logmaxn];     // anc[p][i]是结点p的第2^i级父亲。anc[i][0] = fa[i]
	int maxcost[maxn][logmaxn]; // maxcost[p][i]是i和anc[p][i]的路径上的最大费用
 
	// 预处理，根据fa和cost数组求出anc和maxcost数组
	void preprocess() {
		for(int i = 0; i < n; i++) {
			anc[i][0] = fa[i];
			maxcost[i][0] = cost[i];
			for(int j = 1; (1<= 0; i--)
			if (L[p] - (1 << i) >= L[q]) {
				ans = max(ans, maxcost[p][i]);
				p = anc[p][i];
			}
 
		if (p == q) return ans; // LCA为p
 
		for(int i = log; i >= 0; i--)
			if(anc[p][i] != -1 && anc[p][i] != anc[q][i]) {
				ans = max(ans, maxcost[p][i]);
				p = anc[p][i];
				ans = max(ans, maxcost[q][i]);
				q = anc[q][i];
			}
 
		ans = max(ans, cost[p]);
		ans = max(ans, cost[q]);
		return ans; // LCA为fa[p]（它也等于fa[q]）
	}
};
 
 
// 固定根的最小树型图，邻接矩阵写法
struct MDST {
	int n;
	int w[maxn][maxn]; // 边权
	int vis[maxn];     // 访问标记，仅用来判断无解
	int ans;           // 计算答案
	int removed[maxn]; // 每个点是否被删除
	int cid[maxn];     // 所在圈编号
	int pre[maxn];     // 最小入边的起点
	int iw[maxn];      // 最小入边的权值
	int max_cid;       // 最大圈编号
 
	void init(int n) {
		this->n = n;
		for(int i = 0; i < n; i++)
			for(int j = 0; j < n; j++) w[i][j] = INF;
	}
 
	void AddEdge(int u, int v, int cost) {
		w[u][v] = min(w[u][v], cost); // 重边取权最小的
	}
 
	// 从s出发能到达多少个结点
	int dfs(int s) {
		vis[s] = 1;
		int ans = 1;
		for(int i = 0; i < n; i++)
			if(!vis[i] && w[s][i] < INF) ans += dfs(i);
		return ans;
	}
 
	// 从u出发沿着pre指针找圈
	bool cycle(int u) {
		max_cid++;
		int v = u;
		while(cid[v] != max_cid) {
			cid[v] = max_cid;
			v = pre[v];
		}
		return v == u;
	}
 
	// 计算u的最小入弧，入弧起点不得在圈c中
	void update(int u) {
		iw[u] = INF;
		for(int i = 0; i < n; i++)
			if(!removed[i] && w[i][u] < iw[u]) {
				iw[u] = w[i][u];
				pre[u] = i;
			}
	}
 
	// 根结点为s，如果失败则返回false
	bool solve(int s) {
		memset(vis, 0, sizeof(vis));
		if(dfs(s) != n) return false;
 
		memset(removed, 0, sizeof(removed));
		memset(cid, 0, sizeof(cid));
		for(int u = 0; u < n; u++) update(u);
		pre[s] = s;
		iw[s] = 0; // 根结点特殊处理
		ans = max_cid = 0;
		for(;;) {
			bool have_cycle = false;
			for(int u = 0; u < n; u++) if(u != s && !removed[u] && cycle(u)) {
				have_cycle = true;
				// 以下代码缩圈，圈上除了u之外的结点均删除
				int v = u;
				do {
					if(v != u) removed[v] = 1;
					ans += iw[v];
					// 对于圈外点i，把边i->v改成i->u（并调整权值）；v->i改为u->i
					// 注意圈上可能还有一个v'使得i->v'或者v'->i存在，因此只保留权值最小的i->u和u->i
					for(int i = 0; i < n; i++) if(cid[i] != cid[u] && !removed[i]) {
							if(w[i][v] < INF) w[i][u] = min(w[i][u], w[i][v]-iw[v]);
							w[u][i] = min(w[u][i], w[v][i]);
							if(pre[i] == v) pre[i] = u;
						}
					v = pre[v];
				} while(v != u);
				update(u);
				break;
			}
			if(!have_cycle) break;
		}
		for(int i = 0; i < n; i++)
			if(!removed[i]) ans += iw[i];
		return true;
	}
};
 
 
// 以下为网络流
 
struct Dinic {
	int n, m, s, t;
	int first[maxm], next[maxm];
	Edge edges[maxm];
	bool vis[maxn];
	int d[maxn], cur[maxn];
 
	void init(int n, int s, int t) {
		this->n = n; this->m = 1;
		this->s = s; this->t = t;
	}
	
	void AddEdge(int u, int v, int cap) {
		edges[++m] = (Edge){u, v, cap, 0};  next[m] = first[u]; first[u] = m;
		edges[++m] = (Edge){v, u, 0, 0}; next[m] = first[v]; first[v] = m;
	}
 
	bool BFS() {
		memset(vis, 0, sizeof(vis));
		queue Q;
		Q.push(s);
		vis[s] = 1; d[s] = 0;
		while(!Q.empty()) {
			int u = Q.front(); Q.pop();
			for(int i = first[u]; i; i = next[i]) {
				Edge& e = edges[i];
				if(!vis[e.to] && e.cap > e.flow) {
					vis[e.to] = 1;
					d[e.to] = d[u] + 1;
					Q.push(e.to);
				}
			}
		}
		return vis[t];
	}
 
	int DFS(int u, int a) {
		if(u == t || a == 0) return a;
		int flow = 0, f;
		for(int& i = cur[u]; i; i = next[i]) {
			Edge& e = edges[i];
			if(d[u] + 1 == d[e.to] && (f = DFS(e.to, min(a, e.cap-e.flow))) > 0) {
				edges[i].flow += f;
				edges[i^1].flow -= f;
				flow += f;
				a -= f;
				if(a == 0) break;
			}
		}
		return flow;
	}
 
	int Maxflow() {
		int flow = 0;
		while(BFS()) {
			for(int u = 0; u < n; u++) cur[u] = first[u];
			flow += DFS(s, INF);
		}
		return flow;
	}
};
 
 
struct MCMF {
	int n, m, s, t;
	Edge edges[maxm];
	int first[maxn], next[maxn];
	int inq[maxn], d[maxn], p[maxn], a[maxn];
 
	void init(int n, int s, int t) {
		this->n = n; this->m = 1;
		this->s = s; this->t = t;
	}
 
	void AddEdge(int u, int v, int cap, int cost) {
		edges[++m] = (Edge){u, v, cap, 0, cost}; next[m] = first[u]; first[u] = m;
		edges[++m] = (Edge){v, u, 0, 0, -cost};  next[m] = first[v]; first[v] = m;
	}
 
	bool BellmanFord(int& cost) {
		for(int i = 0; i < n; i++) d[i] = INF;
		memset(inq, 0, sizeof(inq));
		d[s] = 0; inq[s] = 1; p[s] = 0; a[s] = INF;
 
		queue Q;
		Q.push(s);
		while(!Q.empty()) {
			int u = Q.front(); Q.pop();
			inq[u] = 0;
			for(int i = first[u]; i; i = next[i]) {
				Edge& e = edges[i];
				if(e.cap > e.flow && d[e.to] > d[u] + e.cost) {
					d[e.to] = d[u] + e.cost;
					p[e.to] = i;
					a[e.to] = min(a[u], e.cap - e.flow);
					if(!inq[e.to]) { Q.push(e.to); inq[e.to] = 1; }
				}
			}
		}
		if(d[t] == INF) return false;
		cost += d[t] * a[t];
		int u = t;
		while(u != s) {
			edges[p[u]].flow += a[t];
			edges[p[u]^1].flow -= a[t];
			u = edges[p[u]].from;			
		}
		return true;
	}
 
	int Mincost() {
		int cost = 0;
		while(BellmanFord(cost));
		return cost;
	}
};


// 扩展欧几里德算法
 
void gcd(lli a, lli b, lli&d, lli&x, lli&y) {
	if(!b) d = a, x = 1, y = 0;
	else gcd(b, a%b, d, y, x), y -= x*(a/b);
}
 
 
// 逆元
// a 与 n 互质
 
lli inv(lli a, lli n) {
	lli d, x, y;
	gcd(a, n, d, x, y);
	return d == 1 ? (x%n+n)%n : -1;
}
 
 
// 组合数
// 预处理阶乘和逆元会更快
// 还可以用 lucas 定理优化
 
lli C(lli n, lli m, lli p) {
	lli s1 = 1, s2 = 1;
	for(int i = n-m+1; i <= n; i++)
		s1 = s1 * i % p;
	for(int i = 1; i <= m; i++)
		s2 = s2 * i % p;
	return s1 * inv(s2, p) % p;
}
 
 
lli pow_mod(lli a, lli n, lli p) {
	lli ret = 1;
	for(; n; n>>=1, a = a*a % p)
	    if(n & 1) ret = ret*a % p
	return ret;
}
 
 
// 防止乘法溢出
 
lli mul_mod(lli a, lli n, lli p) {
	lli ret = 0;
	for(; n; n>>=1, a = (a<<1) % p) 
		if(n & 1) ret = (ret+a) % p;
	return ret;
}
 
 
// BSGS
// a^x ≡ b (mod p)
// x = k*m + n, n < m
// a^(k*m+n) ≡ b (mod p) => a^(k*m) * a^n ≡ b(mod p)
// 预处理 a^n
// 主过程a^n ≡ b*inv(a^(k*m)) (mod p)
// 预处理 inv(a^m), 则 inv(a^(k*m)) = inv(a^((k-1)*m)) * inv(a^m)
 
// 预处理a^n : O(m*log(m)), 主过程共 p/m 轮 : O(p/m * log(m)).
// 总时间复杂度 : O(m*log(m) + p/m * log(m)) = O((m+p/m)*log(m)), 当 m = sqrt(p) 时总时间较短
 
lli log_mod(lli a, lli b) {
	lli m, v, e = 1;
	m = (lli)(sqrt(p) + 0.5); // notice
	v = pow_mod(pow_mod(a, m), p-2);
	map  x;
	x[1] = 0;
	// 预处理 a^m
	for(int i = 1; i < m; i++) {
		e = e*a % p;
		if(!x.count(e)) x[e] = i;
	}
	for(int i = 0; i < m; i++) {
		if(x.count(b)) return i*m + x[b];
		b = b*v % p;
	}
	return -1;
}
 
 
int vis[maxn];
int prime[maxp];
 
void sieve(int n) {
	int m = (int)(sqrt(n) + 0.5);
	memset(vis, 0, sizeof(vis));
	for(int i = 2; i <= m; i++) if(!vis[i])
		for(int j = i*i; j <= n; j+=i) vis[j] = 1;
}
 
 
int gen_primes(int n) {
	sieve(n);
	int c = 0;
	for(int i = 2; i <= n; i++) if(!vis[i])
		prime[c++] = i;
	return c;
}
 
 
struct Matrix {
	int n, m;
	int a[maxn][maxn];
	
	Matrix(int n=0, int m=0) : n(n), m(m) { memset(a, 0, sizeof(a)); }
	
	// 快速幂
	Matrix operator ^ (int p) {
		Matrix t = Matrix(n, n), A = *this;
		for(int i = 0; i < n; i++) t.a[i][i] = 1;
		for(; p; p>>=1, A=A*A) if(p & 1) t = t*A;
		return t;
	}
	
	Matrix operator * (Matrix A) {
		Matrix ret = Matrix(n, A.m);
		for(int i = 0; i < n; i++)
			for(int j = 0; j < A.m; j++) {
				ret.a[i][j] = 0;
				for(int k = 0; k < m; k++)
					ret.a[i][j] = ((lli)a[i][k]*A.a[k][j] + ret.a[i][j]) % mod;
			}
		return ret;
	}
	
	void print() {
		for(int i = 0; i < n; i++) {
			for(int j = 0; j < m; j++)
				printf("%d ", a[i][j]);
			printf("\n");
		}
	}
};
 
 
struct XorGauss {
	bitset maxn;
	int n;
	
	int gaussJordan() {
		int i, j, k, r = 0;
		for(j = 0; j < n; j++) {
			i = r;
			while(i < m && !A[i][j]) i++;
			if(i >= m) return -1;
			for(k = 0; k < m; k++)
				if(k != i && A[k][j]) A[k]^=A[i];
			swap(A[i], A[r++]);
		}
		return 1;
	}
};
 
 
struct Gauss {
	int n;
	double A[maxn][maxn];
	
	void gauss() {
		int i, j, k, r;
		for(i = 0; i < n; i++) {
			r = i;
			for(j = i+1; j < n; j++)
				if(fabs(A[j][i]) > fabs(A[r][i])) r = j;
			if(r != i) for(j = 0; j <= n; j++) swap(A[r][j], A[i][j]);
			for(k = i+1; k < n; k++) {
				double f = A[k][i] / A[i][i];
				for(j = i; j <= n; j++) A[k][j] -= f * A[i][j];
			}
		}
		for(i = n-1; i >= 0; i--) {
			for(j = i+1; j < n; j++)
				A[i][n] -= A[j][n] * A[i][j];
			A[i][n] /= A[i][i];
		}
	}
	
	void gaussJordan() {
		int i, j, k, r;
		for(i = 0; i < n; i++) {
			r = i;
			for(j = i+1; j < n; j++)
				if(fabs(A[j][i]) > fabs(A[r][i])) r = j;
			if(fabs(A[j][i]) < eps) continue;
			if(r != i) for(j = 0; j <= n; j++) swap(A[r][j], A[i][j]);
			for(k = 0; k < n; k++) if(k != i)
				for(j = n; j >= i; j--) A[k][j] -= A[k][i]/A[i][i] * A[i][j];
		}
	}
};
 
 
struct MU {
	int c, mu[maxn], prime[maxn];
	bool vis[maxn];
	
	void get_mu() {
		mu[1] = 1;
		for(int i = 2; i < maxn; i++) {
			if(!vis[i]) prime[++c] = i, mu[i] = -1;
			for(int j = 1; prime[j]*i < maxn; j++) {
				int k = prime[j] * i;
				vis[k] = 1;
				if(i % prime[j] == 0) { mu[k] = 0; break; }
				mu[k] = -mu[i];
			}
		}
	}
};
 
 
struct Simpson {
	double F(double x) {}
	
	double simpson(double a, double b) {
		double c = a + (b-a)/2;
		return (F(a)+4*F(c)+F(b))*(b-a)/6;
	}
	
	// 保证误差 < eps
	double asr(double a, double b, double eps, double A) {
		double c = a + (b-a)/2;
		double L = simpson(a, c), R = simpson(c, b);
		if(fabs(L+R-A) <= 15.0*eps) return L+R+(L+R-A)/15.0; // 这个 15.0 也是用来调整精度的
		return asr(a, c, eps/2, L) + asr(c, b, eps/2, R);
	}
	
	// 保证误差 < eps
	double asr(double a, double b, double eps) {
		return asr(a, b, eps, simpson(a, b));
	}
};
 
 
struct EulerPhi {
	int phi[maxn];
	
	void euler_phi(int n) {
		phi[1] = 1;
		for(int i = 2; i <= n; i++) if(!phi[i])
			for(int j = i; j <= n; j += i) {
				if(!phi[j]) phi[j] = j;
				phi[j] = phi[j] / i * (i-1);
			}
	}
	
	int get_phi(int n) {
		int ret = n;
		int m = (int)(sqrt(n) + 0.5);
		for(int i = 2; i <= m; i++) if(n % i == 0) {
			ret = ret / i * (i-1);
			while(n % i == 0) n /= i;
		}
		if(n > 1) ret = ret / n * (n-1);
		return ret;
	}
};
 
 
// Catalan
// h(0) = h(1) = 1
// h(n) = h(0)*h(n-1) + h(1)*h(n-2) + ... + h(n-1)h(0) (n>=2)
 
int h[maxn];
 
void Catalan(int n) {
	h[0] = h[1] = 1;
	for(int i = 2; i <= n; i++)
		for(int j = 0; j < i; j++)
			h[i] += h[j] * h[i-j];
	// n >= 22, beyond
}
 
 
// 斯特灵数
 
int s1[maxn][maxn];
int s2[maxn][maxn];
 
// 第一类: n个元素分成k个非空循环排列(环)的方法总数
// s(n, k) = s(n-1, k-1) + (n-1)*s(n-1, k)
 
// 第二类: n个元素放到k个集合内的方法总数
// s(n, k) = s(n-1, k-1) + k*s(n-1, k)
 
void Stirling1(int n) {
	s1[1][1] = 1;
	for(int i = 2; i <= n; i++)
		for(int j = 1; j <= i; j++)
			s1[i][j] = s1[i-1][j-1] + (i-1) * s1[i-1][j];
}
 
void Stirling2(int n) {
	s2[0][0] = 1;
	for(int i = 1; i <= n; i++) {
		s2[i][1] = 1;
		for(int j = 2; j <= i; j++)
			s2[i][j] = s2[i-1][j-1] + j * s2[i-1][j];
	}
}


struct Point {
	double x, y;
	Point(double x = 0, double y = 0): x(x), y(y) {}
};
 
typedef Point Vector;
 
Vector operator + (Vector A, Vector B) { return Vector(A.x+B.x, A.y+B.y); }
Vector operator - (Vector A, Vector B) { return Vector(A.x-B.x, A.y-B.y); }
Vector operator * (Vector A, double p) { return Vector(A.x*p, A.y*p); }
Vector operator / (Vector A, double p) { return Vector(A.x/p, A.y/p); }
 
bool operator < (const Vector& a, const Vector& b) {
	return a.x < b.x || (a.x == b.x && a.y < b.y);
}
 
const double eps = 1e-10;
 
int dcmp(double x) {
	if(fabs(x) < eps) return 0;
	else return x < 0 ? -1 : 1;
}
 
bool operator == (const Vector& a, const Vector& b) {
	return dcmp(a.x-b.x) == 0 && dcmp(a.y-b.y) == 0;
}
 
double Dot(Vector A, Vector B) { return A.x*B.x + A.y*B.y; }
double Length(Vector A) { return sqrt(Dot(A, A)); }
double Angle(Vector A, Vector B) { return acos(Dot(A, B) / Length(A) / Length(B)); }
double Cross(Vector A, Vector B) { return A.x*B.y - A.y*B.x; }
double Area2(Point A, Point B, Point C) { return Cross(B-A, C-A); }
 
// 旋转公式 用 (length*cos(theta), length*sin(theta)) 表示向量即可推出
Vector Rotate(Vector A, double rad) {
	return Vector(A.x*cos(rad)-A.y*sin(rad), A.x*sin(rad)+A.y*cos(rad));
}
 
// 法向量
Vector Normal(Vector A) {
	double L = Length(A);
	return Vector(-A.y/L, A.x/L);
}
 
// P(x1, y1), v(x2, y2), Q(x3, y3), w(x4, y4), P-Q=u(x5, y5).
// P + v*t1 = Q + w*t2 // 用虚数表示如下
// x1+y1*i + t1*x2+t1*y2*i = x3+y3*i + t2*x4+t2*y4*i
// x1 + t1*x2 + (y1+t1*y2)*i = x3 + t2*x4 + (y3+t2*y4)*i
 
// => /x1+t1*x2 = x3+t2*x4 -> x5 + t1*x2 = t2*x4 -> x5*y4 + t1*x2*y4 = t2*x4*y4 (1)
//    \y1+t1*y2 = y3+t2*y4 -> y5 + t1*y2 = t2*y4 -> y5*x4 + t1*y2*x4 = t2*y4*x4 (2)
 
// (1)-(2) => (x5*y4-y5*x4) + t1(x2*y4-y2*x4) = 0
// => t1 = (x4*y5-x5*y4) / (x2*y4-y2*x4)
//       = Cross(w, u) / Cross(v, w)
 
Vector GetLineIntersection(Point P, Vector v, Point Q, Vector w) {
	Vector u = P-Q;
	double t = Cross(w, u) / Cross(v, w);
	return P+v*t;
}
 
double DistanceToLine(Point P, Point A, Point B) {
	Vector v1 = B-A, v2 = P-A;
	return fabs(Cross(v1, v2)) / Length(v1);
}
 
double DistanceToSegment(Point P, Point A, Point B) {
	if(A == B) return Length(P-A); // 两点
	Vector v1 = B-A, v2 = P-A, v3 = P-B;
	if(dcmp(Dot(v1, v2)) < 0) return Length(v2);
	if(dcmp(Dot(v1, v3)) > 0) return Length(v3);
	return fabs(Cross(v1, v2) / Length(v1));
}
 
Point GetLineProjection(Point P, Point A, Point B) {
	Vector v = B-A;
	return A+v*(Dot(v, P-A) / Dot(v, v));
}
 
bool SegmentProperIntersection(Point a1, Point a2, Point b1, Point b2) {
	double c1=Cross(a2-a1,b1-a1), c2=Cross(a2-a1,b2-a1), c3=Cross(b2-b1,a1-b1), c4=Cross(b2-b1,a2-b1);
	return (dcmp(c1)*dcmp(c2) < 0) && (dcmp(c3)*dcmp(c4) < 0);
}
 
bool OnSegment(Point P, Point a1, Point a2) {
	return dcmp(Cross(a1-P, a2-P)) == 0 && dcmp(Dot(a1-P, a2-P)) < 0;
}
 
double PolygonArea(Point* P, int n) {
	double area = 0;
	for(int i = 1; i < n-1; i++)
		area += Cross(P[i]-P[0], P[i+1]-P[0]);
	return area/2; // 有向面积
}
 
struct Line {
	Point p;
	Vector v;
	Point point(double a) {
		return p + v*a;
	}
};
 
const double PI = acos(double(-1));
 
struct Circle {
	Point c;
	double r;
	Circle(Point c, double r):c(c),r(r) {}
	Point point(double a) {
		return Point(c.x + cos(a)*r, c.y + sin(a)*r);
	}
};
 
int getLineCircleIntersection(Line L, Circle C, double& t1, double& t2, vector& sol) {
	double a = L.v.x, b = L.p.x - C.c.x, c = L.v.y, d = L.p.y - C.c.y;
	double e = a*a + c*c, f = 2 * (a*b+c*d), g = b*b + d*d - C.r*C.r;
	double delta = f*f - 4*e*g;
	if(dcmp(delta) < 0) return 0;
	if(dcmp(delta) == 0) {
		t1 = t2 = -f / (2*e);
		sol.push_back(L.point(t1));
		return 1;
	}
	t1 = (-f - sqrt(delta)) / (2*e);
	sol.push_back(L.point(t1));
	t2 = (-f + sqrt(delta)) / (2*e);
	sol.push_back(L.point(t2));
	return 2;
}
 
double angle(Vector v) {
	return atan2(v.y, v.x);
}
 
int getCircleCircleIntersection(Circle C1, Circle C2, vector& sol) {
	double d = Length(C1.c - C2.c);
	if(dcmp(d) == 0) {
		if(dcmp(C1.r-C2.r) == 0) return -1;
		return 0;
	}
	if(dcmp(C1.r+C2.r-d) < 0) return 0;
	if(dcmp(fabs(C1.r-C2.r)-d > 0)) return 0;
 
	double a = angle(C2.c-C1.c);
	double da = acos((C1.r*C1.r + d*d - C2.r*C2.r) / (2*C1.r*d));
 
	Point p1 = C1.point(a-da), p2 = C1.point(a+da);
	sol.push_back(p1);
	if(p1 == p2) return 1;
	sol.push_back(p2);
	return 2;
}
 
int getTangents(Point p, Circle C, Vector* v) {
	Vector u = C.c-p;
	double dist = Length(u);
	if(dist < C.r) return 0;
	else if(dcmp(dist-C.r) == 0) {
		v[0] = Rotate(u, PI/2);
		return 1;
	} else {
		double ang = asin(C.r / dist);
		v[0] = Rotate(u, -ang);
		v[1] = Rotate(u, +ang);
		return 2;
	}
}
 
// 可用 dcmp 比较
// 如果不希望在凸包边上有输入点, 可以将 <= 换为 <
int ConvexHull(Point* p, Point* ch, int n) {
	sort(p, p+n); // x 为第一关键字, y 为第二关键字
	int m = 0;
	for(int i = 0; i < n; i++) {
		while(m > 1 && Cross(ch[m-1]-ch[m-2], p[i]-ch[m-2]) <= 0) m--;
		ch[m++] = p[i];
	}
	int k = m;
	for(int i = n-2; i >= 0; i--) {
		while(m > k && Cross(ch[m-1]-ch[m-2], p[i]-ch[m-2]) <= 0) m--;
		ch[m++] = p[i];
	}
	if(n > 1) m--;
	return m;
}
 
bool OnLeft(const Line& L, const Point& p) {
	return Cross(L.v, p-L.P) > 0;
}
 
vector HalfplaneIntersection(vector L) {
	int n = L.size();
	sort(L.begin(), L.end()); // 按极角排序
 
	int first, last;          // 双端队列的第一个元素和最后一个元素的下标
	vector p(n);       // p[i]为q[i]和q[i+1]的交点
	vector q(n);        // 双端队列
	vector ans;        // 结果
 
	q[first=last=0] = L[0];  // 双端队列初始化为只有一个半平面L[0]
	for(int i = 1; i < n; i++) {
		while(first < last && !OnLeft(L[i], p[last-1])) last--;
		while(first < last && !OnLeft(L[i], p[first])) first++;
		q[++last] = L[i];
		if(fabs(Cross(q[last].v, q[last-1].v)) < eps) { // 两向量平行且同向，取内侧的一个
			last--;
			if(OnLeft(q[last], L[i].P)) q[last] = L[i];
		}
		if(first < last) p[last-1] = GetLineIntersection(q[last-1], q[last]);
	}
	while(first < last && !OnLeft(q[first], p[last-1])) last--; // 删除无用平面
	if(last - first <= 1) return ans; // 空集
	p[last] = GetLineIntersection(q[last], q[first]); // 计算首尾两个半平面的交点
 
	// 从deque复制到输出中
	for(int i = first; i <= last; i++) ans.push_back(p[i]);
	return ans;
}
 
int diameter2(vector& points) {
	vector p = ConvexHull(points);
	int n = p.size();
	if(n == 1) return 0;
	if(n == 2) return Dist2(p[0], p[1]);
	p.push_back(p[0]); // 免得取模
	int ans = 0;
	for(int u = 0, v = 1; u < n; u++) {
		// 一条直线贴住边p[u]-p[u+1]
		while(1) {
			// 当 Area(p[u], p[u+1], p[v+1]) <= Area(p[u], p[u+1], p[v])时停止旋转
			// 即 Cross(p[u+1]-p[u], p[v+1]-p[u]) - Cross(p[u+1]-p[u], p[v]-p[u]) <= 0
			// 根据 Cross(A,B) - Cross(A,C) = Cross(A,B-C)
			// 化简得 Cross(p[u+1]-p[u], p[v+1]-p[v]) <= 0
			int diff = Cross(p[u+1]-p[u], p[v+1]-p[v]);
			if(diff <= 0) {
				ans = max(ans, Dist2(p[u], p[v])); // u和v是对踵点
				if(diff == 0) ans = max(ans, Dist2(p[u], p[v+1])); // diff == 0时u和v+1也是对踵点
				break;
			}
			v = (v + 1) % n;
		}
	}
	return ans;
}

