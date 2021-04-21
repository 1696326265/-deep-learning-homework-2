#include<bits/stdc++.h>
using namespace std;

#define poly vector<int>
#define ll long long

const int mod=998244353, gen=3;

ll K(ll x,ll y=mod-2){
	ll t=1;
	for (;y;y>>=1,x=x*x%mod)
		if (y&1) t=t*x%mod;
	return t;
}
inline int fix(int x){
	return x<mod? x: x-mod;
}

void dft(poly &_a,int n){
	_a.resize(1<<n); int *a=&(_a.front());
	static vector<int>dp; dp.resize(1<<n);
	for (int i=1;i<(1<<n);++i){
		dp[i]=i&1? dp[i^1]|1<<(n-1): dp[i>>1]>>1;
		if (dp[i]>i) swap(a[dp[i]],a[i]);
	}
	for (int b=0;b<n;++b){
		int len=1<<b;
		static vector<int>W; W.resize(len);
		{
			ll w=1, w0=K(gen,(mod-1)/len/2);
			for (int j=0;j<len;++j) W[j]=w, w=w*w0%mod;
		}
		for (int i=0;i<(1<<n);i+=len*2){
			int *l=a+i, *r=a+i+len;
			for (int j=0;j<len;++j){
				int x=l[j], y=(ll)r[j]*W[j]%mod;
				l[j]=fix(x+y); r[j]=fix(x-y+mod);
			}
		}
	}
}
void idft(poly &a,int n){
	a.resize(1<<n); reverse(a.begin()+1,a.end()); dft(a,n);
	ll inv=K(1<<n); for (auto &o:a) o=o*inv%mod;
}

int main(){
	int n, m;
	poly a, b;
	cin>>n>>m; a.resize(n+1); b.resize(m+1);
	for (auto &o:a) cin>>o;
	for (auto &o:b) cin>>o;
	dft(a,18); dft(b,18);
	for (int i=0;i<a.size();++i) a[i]=a[i]*(ll)b[i]%mod;
	idft(a,18); a.resize(n+m+1);
	for (auto o:a) cout<<o<<' ';
	puts("");
}