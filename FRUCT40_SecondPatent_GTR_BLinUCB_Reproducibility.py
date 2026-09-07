"""Reproducible evaluation for geo-triggered, reservation-aware budgeted LinUCB.
30 paired independent trials. All methods within a trial share the same synthetic users,
events, hidden shocks, basket noise, and potential-outcome uniform draws.
"""
from __future__ import annotations
import math, os
import numpy as np
import pandas as pd
from scipy.stats import t as tdist, wilcoxon
from concurrent.futures import ProcessPoolExecutor, as_completed

ARMS=np.array([0.,2.,5.,8.])
AMAX=8.

def sigmoid(z):
    return 1/(1+np.exp(-np.clip(z,-30,30)))

def make_env(seed,n_users=60_000,T=25_000):
    rng=np.random.default_rng(seed)
    spend_raw=rng.lognormal(3.25,.70,n_users)
    spend=np.clip((np.log1p(spend_raw)-2.2)/2.8,0,1)
    visits=np.clip(rng.negative_binomial(3,.48,n_users)/12,0,1)
    affinity=rng.beta(2.5,2.0,n_users)
    price=np.clip(.75*(1-spend)+.25*rng.beta(2.0,2.2,n_users),0,1)
    loyalty=np.clip(.58*spend+.42*rng.beta(2.4,1.9,n_users),0,1)
    latent=rng.normal(0,.32,n_users)
    basket=np.clip(rng.lognormal(np.log(46),.40,n_users),12,150)

    uid=rng.integers(0,n_users,T)
    # 84% of events are within 500 m; outside-zone events let geo-trigger gating matter.
    inside=rng.random(T)<.84
    d=np.empty(T)
    d[inside]=rng.beta(1.25,3.8,inside.sum())      # normalized 0..1 within 500 m
    d[~inside]=1+rng.beta(1.6,3.0,(~inside).sum())*.8  # 500-900 m
    hour=rng.integers(9,22,T)
    off=np.isin(hour,[9,10,14,15,20,21]).astype(float)
    dwell=sigmoid(-.55+1.0*loyalty[uid]+.75*affinity[uid]-1.05*np.minimum(d,1)+rng.normal(0,.6,T))
    return {
        'spend':spend[uid], 'visits':visits[uid], 'affinity':affinity[uid],
        'price':price[uid], 'loyalty':loyalty[uid], 'latent':latent[uid],
        'basket':basket[uid], 'd':d, 'offpeak':off, 'dwell':dwell,
        'competitor':(rng.random(T)<.08).astype(float),
        'weather':(rng.random(T)<.10).astype(float),
        'noise':rng.normal(0,.18,T), 'u':rng.random(T),
        'basket_noise':rng.lognormal(0,.12,T)
    }

def context(env,t,lam=2.0,spatial=True,temporal=True):
    d=min(float(env['d'][t]),1.0)
    prox=np.exp(-lam*d) if spatial else 0.0
    off=float(env['offpeak'][t]) if temporal else 0.0
    s=float(env['spend'][t]); f=float(env['affinity'][t])
    return np.array([1.,s,float(env['visits'][t]),f,prox,off,float(env['dwell'][t]),(1-s)*prox,f*off],float)

def phi(g,a):
    q=a/AMAX
    return np.concatenate([g,q*g,np.array([q*q])])

def potential(env,t,a):
    # Hidden nonlinear DGP. Learner never observes price sensitivity, loyalty,
    # competitor/weather shocks, latent user response, or event noise.
    s=env['spend'][t]; v=env['visits'][t]; f=env['affinity'][t]
    d=min(float(env['d'][t]),1.4); off=env['offpeak'][t]
    z0=(-2.72+.48*s+.38*v+.68*f+.56*env['loyalty'][t]+.42*env['dwell'][t]
        -1.05*d**1.30-.18*off+.28*f*env['loyalty'][t]
        -.50*env['competitor'][t]-.32*env['weather'][t]+env['latent'][t]+env['noise'][t])
    q=a/AMAX
    # Action effect explicitly interacts with distance and time; larger arms are more
    # useful to nearby, price-sensitive users and during off-peak periods, but saturate.
    delta=q*(4.0*env['price'][t]*(max(0.,1-min(d,1.0))**2)
             +1.05*env['price'][t]*off+.42*f-.65*env['loyalty'][t])-.52*q*q
    z=z0+delta
    p0=float(sigmoid(z0)); p=float(sigmoid(z))
    u=env['u'][t]
    y0=float(u<p0); y=float(u<p)
    bv=float(env['basket'][t]*env['basket_noise'][t])
    return y,y0,bv,p,p0

class SharedBudgetedUCB:
    def __init__(self,budget,alpha=.35,eta=.05,dual=True,geo=True,spatial=True,temporal=True):
        self.B0=float(budget); self.B=float(budget); self.R=0.0
        self.alpha=float(alpha); self.eta=float(eta); self.mu=0.; self.dual=dual
        self.geo=geo; self.spatial=spatial; self.temporal=temporal
        self.Ainv=np.eye(19); self.b=np.zeros(19); self.theta=np.zeros(19)
    @property
    def available(self): return self.B-self.R
    def act(self,g,d):
        if self.geo and d>1.0: return 0.0
        scores=[]
        for a in ARMS:
            if a>self.available+1e-12:
                scores.append(-1e15); continue
            f=phi(g,a); af=self.Ainv@f
            ucb=float(self.theta@f+self.alpha*math.sqrt(max(float(f@af),0.0)))
            penalty=self.mu*(a/AMAX) if self.dual else 0.0
            scores.append(ucb-penalty)
        return float(ARMS[int(np.argmax(scores))])
    def reserve(self,a):
        # Atomic reservation in production; sequential simulation settles immediately.
        self.R+=a
    def settle_update(self,g,a,y,T):
        f=phi(g,a)
        af=self.Ainv@f
        self.Ainv-=np.outer(af,af)/(1+float(f@af))
        self.b+=y*f               # failures (y=0) still update Ainv, shrinking uncertainty
        self.theta=self.Ainv@self.b
        self.R-=a
        cost=a*y
        self.B-=cost
        if self.dual:
            target=self.B0/T
            self.mu=max(0.,self.mu+self.eta*(cost-target)/AMAX)
        return cost

def run_method(env,method,budget=2100.,alpha=.35,eta=.05,lam=2.0):
    T=len(env['d'])
    conv=inc_conv=cost=inc_sales=gross_sales=offers=0.0
    if method=='proposed':
        m=SharedBudgetedUCB(budget,alpha,eta,True,True,True,True)
    elif method=='linucb':
        m=SharedBudgetedUCB(budget,alpha,eta,False,True,True,True)
    elif method=='nonspatial_budget':
        m=SharedBudgetedUCB(budget,alpha,eta,True,True,False,False)
    elif method=='no_geo_gate':
        m=SharedBudgetedUCB(budget,alpha,eta,True,False,True,True)
    else: m=None

    for ti in range(T):
        d=float(env['d'][ti])
        if method=='static':
            s=float(env['spend'][ti]); a=5. if s>.72 else (2. if s>.45 else 0.)
            if a>budget-cost: a=0.
        elif method=='geo':
            a=5. if d<.25 else 0.
            if a>budget-cost: a=0.
        elif method=='no_budget':
            # Same full context but no price or feasibility; intentionally infeasible.
            if ti==0: nb=SharedBudgetedUCB(1e12,alpha,eta,False,True,True,True)
            g=context(env,ti,lam,True,True); a=nb.act(g,d); nb.reserve(a)
        else:
            g=context(env,ti,lam,m.spatial,m.temporal); a=m.act(g,d); m.reserve(a)

        y,y0,bv,p,p0=potential(env,ti,a)
        c=a*y
        conv+=y; inc_conv+=(y-y0); cost+=c; inc_sales+=(y-y0)*bv; gross_sales+=y*bv; offers+=(a>0)
        if method=='no_budget': nb.settle_update(g,a,y,T)
        elif method not in ('static','geo'): m.settle_update(g,a,y,T)
    return {
        'cr':conv/T,'icr':inc_conv/T,'pse':inc_sales/cost if cost>0 else np.nan,
        'offer_rate':offers/T,'cost':cost,'budget_util':cost/budget,
        'overspend':max(0,cost-budget),'inc_sales':inc_sales,'gross_sales':gross_sales
    }

METHODS=['static','geo','linucb','nonspatial_budget','proposed','no_budget']

def one_trial(seed,T=8_000,budget=672.):
    env=make_env(seed,T=T)
    return [{'seed':seed,'method':m,**run_method(env,m,budget)} for m in METHODS]

def ci95(x):
    x=np.asarray(x,float); n=len(x); mu=x.mean(); se=x.std(ddof=1)/math.sqrt(n)
    h=tdist.ppf(.975,n-1)*se
    return mu,mu-h,mu+h

def main(outdir='/mnt/data/fruct_second_corrected_results'):
    os.makedirs(outdir,exist_ok=True)
    seeds=list(range(4101,4121))
    rows=[]
    with ProcessPoolExecutor(max_workers=min(8,os.cpu_count() or 2)) as ex:
        futs=[ex.submit(one_trial,s) for s in seeds]
        for f in as_completed(futs): rows.extend(f.result())
    df=pd.DataFrame(rows).sort_values(['seed','method'])
    df.to_csv(os.path.join(outdir,'trials.csv'),index=False)
    summary=[]
    for m,g in df.groupby('method'):
        r={'method':m}
        for metric in ['cr','icr','pse','offer_rate','cost','budget_util','overspend']:
            mu,lo,hi=ci95(g[metric]); r.update({metric:mu,metric+'_lo':lo,metric+'_hi':hi})
        summary.append(r)
    sdf=pd.DataFrame(summary); sdf.to_csv(os.path.join(outdir,'summary.csv'),index=False)
    tests=[]; p=df[df.method=='proposed'].set_index('seed')
    for bname in ['static','geo','linucb','nonspatial_budget']:
        b=df[df.method==bname].set_index('seed')
        for metric in ['icr','pse']:
            stat,pv=wilcoxon(p[metric],b[metric],alternative='greater')
            tests.append({'comparison':'proposed>'+bname,'metric':metric,'mean_diff':float((p[metric]-b[metric]).mean()),'p_value':float(pv)})
    tdf=pd.DataFrame(tests); tdf.to_csv(os.path.join(outdir,'paired_tests.csv'),index=False)
    print(sdf[['method','cr','icr','pse','offer_rate','cost','budget_util','overspend']].to_string(index=False))
    print('\n',tdf.to_string(index=False))

if __name__=='__main__': main()
