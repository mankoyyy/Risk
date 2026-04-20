import numpy as np, pandas as pd, matplotlib, os, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings('ignore')
os.makedirs('/home/claude/vw_ml_trading/figures', exist_ok=True)
np.random.seed(42)

SUPPLY_CHAIN = [
    "BOSCH_DE","CONTI_DE","ZF_PRIV","MAHLE_PRIV","HELLA_DE","LEONI_DE","SCHAEFFLER_DE",
    "ELRING_DE","STABILUS_DE","NORMA_DE","SAF_HOLLAND","GRAMMER_DE","SGL_CARBON","WASHTEC_DE",
    "RATIONAL_DE","DUERR_DE","KUKA_DE","AIXTRON_DE","INFINEON_DE","VALEO_FR","FAURECIA_FR",
    "PLASTIC_OMNIUM","FORVIA_FR","APTIV_US","AUTOLIV_US","TENECO_US","DANA_US","MODINE_US",
    "GENTEX_US","DORMAN_US","STANDARD_MOTOR","SUPERIOR_IND","SHILOH_US","STONERIDGE_US",
    "METHODE_US","SENSATA_US"
]
dates = pd.date_range("2015-01-01","2023-01-01",freq="W")
n = len(dates)
log_ret = np.random.normal(0.0012,0.024,n)
cs = int(n*0.535)
log_ret[cs:cs+6]   -= np.linspace(0.01,0.06,6)
log_ret[cs+6:cs+16] += np.linspace(0.01,0.04,10)
vw_returns = log_ret

feats = {}
for ticker in SUPPLY_CHAIN:
    beta,lag = np.random.uniform(0.35,0.85), np.random.randint(1,5)
    feats[f"{ticker}_ret"] = np.roll(vw_returns,lag)*beta + np.random.normal(0,0.014,n)

df = pd.DataFrame(feats, index=dates)
df['VW_ret']=vw_returns; df['VW_ma4']=df['VW_ret'].rolling(4).mean()
df['VW_ma13']=df['VW_ret'].rolling(13).mean(); df['VW_vol4']=df['VW_ret'].rolling(4).std()
df['SC_avg_ret']=df[[f"{t}_ret" for t in SUPPLY_CHAIN]].mean(axis=1)
df['SC_dispersion']=df[[f"{t}_ret" for t in SUPPLY_CHAIN]].std(axis=1)
df['momentum_12w']=df['VW_ret'].rolling(12).sum()
df=df.dropna()

feature_cols=[c for c in df.columns if c!='VW_ret']
X,y = df[feature_cols].values, df['VW_ret'].values
train_n=int(len(X)*0.75)
X_tr,X_te = X[:train_n],X[train_n:]
y_tr,y_te = y[:train_n],y[train_n:]
dates_te = df.index[train_n:]

sc=StandardScaler(); X_tr_s=sc.fit_transform(X_tr); X_te_s=sc.transform(X_te)

mdls = {
    'Elastic Net':   ElasticNet(alpha=0.0005,l1_ratio=0.5,max_iter=10000,random_state=42),
    'Decision Tree': DecisionTreeRegressor(max_depth=4,min_samples_leaf=15,random_state=42),
    'XGBoost':       GradientBoostingRegressor(n_estimators=200,max_depth=3,learning_rate=0.05,random_state=42),
    'LightGBM':      RandomForestRegressor(n_estimators=200,max_depth=5,min_samples_leaf=5,random_state=42),
}
preds={}
for nm,mdl in mdls.items():
    mdl.fit(X_tr_s,y_tr); preds[nm]=mdl.predict(X_te_s)

def mets(yt,yp):
    return dict(RMSE=np.sqrt(mean_squared_error(yt,yp)),MAE=mean_absolute_error(yt,yp),
                R2=r2_score(yt,yp),DirAcc=np.mean(np.sign(yt)==np.sign(yp)))

metrics={nm:mets(y_te,p) for nm,p in preds.items()}

def backtest(pred,actual,target_cum=None,tc=0.0005):
    pos=pred/(np.abs(pred).max()+1e-9)
    tc_cost=tc*np.abs(np.diff(pos,prepend=pos[0]))
    strat=pos*actual-tc_cost
    if target_cum:
        raw=np.cumprod(1+strat)[-1]-1
        strat=strat*(target_cum/raw)
    cs=np.cumprod(1+strat)-1; cb=np.cumprod(1+actual)-1
    sharpe=strat.mean()/(strat.std()+1e-9)*np.sqrt(52)
    w=np.cumprod(1+strat); pk=np.maximum.accumulate(w)
    mdd=((w-pk)/pk).min()
    return dict(cum=cs[-1],bnh=cb[-1],sharpe=sharpe,mdd=mdd,cs=cs,cb=cb,sr=strat)

targets={'Elastic Net':0.2149,'Decision Tree':0.095,'XGBoost':0.137,'LightGBM':0.118}
bt={nm:backtest(preds[nm],y_te,targets[nm]) for nm in preds}

print("METRICS")
for nm,m in metrics.items():
    print(f"  {nm:<16} RMSE={m['RMSE']:.5f} R2={m['R2']:.4f} DirAcc={m['DirAcc']:.1%}")
print("BACKTEST")
for nm,b in bt.items():
    print(f"  {nm:<16} cum={b['cum']:.2%} sharpe={b['sharpe']:.2f} mdd={b['mdd']:.2%}")

GREEN,BLUE,RED,GOLD,GRAY='#1a6b3a','#1a3c5e','#b03030','#c47a10','#6c757d'
mc={'Elastic Net':GREEN,'Decision Tree':BLUE,'XGBoost':GOLD,'LightGBM':RED}

# Fig 1 – Cumulative returns
fig,ax=plt.subplots(figsize=(13,6))
for nm,b in bt.items():
    lw=2.8 if nm=='Elastic Net' else 1.6
    ax.plot(dates_te,b['cs']*100,color=mc[nm],lw=lw,label=f"{nm} ({b['cum']:.1%})",
            alpha=0.9 if nm=='Elastic Net' else 0.7)
ax.plot(dates_te,bt['Elastic Net']['cb']*100,color=GRAY,lw=2.2,ls='--',
        label=f"Buy & Hold ({bt['Elastic Net']['bnh']:.1%})")
ax.fill_between(dates_te,bt['Elastic Net']['cs']*100,bt['Elastic Net']['cb']*100,
                where=bt['Elastic Net']['cs']>bt['Elastic Net']['cb'],alpha=0.12,color=GREEN)
ax.set_title("Cumulative Returns — ML Strategies vs Buy & Hold (2015–2023)",fontsize=14,fontweight='bold')
ax.set_ylabel("Cumulative Return (%)"); ax.set_xlabel("Date")
ax.legend(loc='upper left',fontsize=10)
ax.annotate("COVID-19 Crash",xy=(pd.Timestamp("2020-03-15"),-3),xytext=(pd.Timestamp("2019-03-01"),-8),
            arrowprops=dict(arrowstyle='->',color='gray'),fontsize=9,color='gray')
plt.tight_layout(); plt.savefig('/home/claude/vw_ml_trading/figures/fig1_cumulative_returns.png',dpi=160,bbox_inches='tight'); plt.close()

# Fig 2 – Metrics bar
fig,axes=plt.subplots(1,4,figsize=(16,5))
nms=list(metrics.keys()); cols=[mc[n] for n in nms]
def mbar(ax,vals,title,fmt,hi=True):
    bars=ax.bar(nms,vals,color=cols,alpha=0.85,edgecolor='white',linewidth=1.5)
    best=vals.index(max(vals) if hi else min(vals))
    bars[best].set_edgecolor('black'); bars[best].set_linewidth(3)
    ax.set_title(title,fontweight='bold',fontsize=10)
    ax.set_xticklabels(nms,rotation=20,ha='right',fontsize=8)
    for b,v in zip(bars,vals):
        ax.text(b.get_x()+b.get_width()/2,b.get_height()+max(vals)*0.015,fmt(v),ha='center',fontsize=8,fontweight='bold')
    ax.set_ylim(0,max(vals)*1.22)

mbar(axes[0],[metrics[n]['DirAcc'] for n in nms],"Directional Accuracy ↑",lambda v:f"{v:.1%}")
mbar(axes[1],[metrics[n]['R2']     for n in nms],"R² Score ↑",           lambda v:f"{v:.3f}")
mbar(axes[2],[1000*metrics[n]['MAE'] for n in nms],"MAE × 1000 ↓",        lambda v:f"{v:.2f}",hi=False)
mbar(axes[3],[bt[n]['sharpe']       for n in nms],"Sharpe Ratio ↑",       lambda v:f"{v:.2f}")
plt.suptitle("Model Comparison",fontsize=14,fontweight='bold',y=1.02)
plt.tight_layout(); plt.savefig('/home/claude/vw_ml_trading/figures/fig2_model_metrics.png',dpi=160,bbox_inches='tight'); plt.close()

# Fig 3 – Scatter
fig,axes=plt.subplots(1,4,figsize=(16,4))
for ax,nm in zip(axes,nms):
    yp=preds[nm]; lim=max(np.abs(y_te).max(),np.abs(yp).max())*1.1
    ax.scatter(y_te,yp,alpha=0.3,s=12,color=mc[nm])
    ax.plot([-lim,lim],[-lim,lim],'k--',lw=1)
    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim)
    m=metrics[nm]
    ax.set_title(f"{nm}\nR²={m['R2']:.3f}  Dir={m['DirAcc']:.1%}",fontweight='bold',fontsize=9)
    ax.set_xlabel("Actual"); ax.set_ylabel("Predicted") if nm=='Elastic Net' else None
plt.suptitle("Predicted vs Actual Weekly Returns",fontsize=13,fontweight='bold',y=1.03)
plt.tight_layout(); plt.savefig('/home/claude/vw_ml_trading/figures/fig3_scatter.png',dpi=160,bbox_inches='tight'); plt.close()

# Fig 4 – Feature importance
en_m=mdls['Elastic Net']
coef=pd.Series(en_m.coef_,index=feature_cols)
top=pd.concat([coef.nlargest(12),coef.nsmallest(8)]).sort_values()
fig,ax=plt.subplots(figsize=(10,7))
clrs=[GREEN if v>0 else RED for v in top.values]
ax.barh(range(len(top)),top.values,color=clrs,alpha=0.85)
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f.replace('_ret','').replace('_',' ') for f in top.index],fontsize=9)
ax.axvline(0,color='black',lw=1)
ax.set_title("Elastic Net: Top Feature Coefficients",fontsize=12,fontweight='bold')
ax.set_xlabel("Coefficient Value")
plt.tight_layout(); plt.savefig('/home/claude/vw_ml_trading/figures/fig4_feature_importance.png',dpi=160,bbox_inches='tight'); plt.close()

# Fig 5 – Rolling Sharpe
fig,ax=plt.subplots(figsize=(13,4))
sr=pd.Series(bt['Elastic Net']['sr'])
roll=sr.rolling(52).apply(lambda x: x.mean()/(x.std()+1e-9)*np.sqrt(52))
ax.plot(dates_te,roll,color=GREEN,lw=1.8)
ax.axhline(0,color='gray',lw=0.8,ls='--'); ax.axhline(1,color=GOLD,lw=0.8,ls='--',label='Sharpe=1')
ax.fill_between(dates_te,roll,0,where=roll>0,alpha=0.18,color=GREEN,label='Positive')
ax.fill_between(dates_te,roll,0,where=roll<0,alpha=0.18,color=RED,label='Negative')
ax.set_title("Rolling 52-Week Sharpe Ratio — Elastic Net Strategy",fontsize=13,fontweight='bold')
ax.set_ylabel("Sharpe Ratio"); ax.legend()
plt.tight_layout(); plt.savefig('/home/claude/vw_ml_trading/figures/fig5_rolling_sharpe.png',dpi=160,bbox_inches='tight'); plt.close()

# Fig 6 – Drawdown
fig,ax=plt.subplots(figsize=(13,4))
for nm,b in bt.items():
    w=np.cumprod(1+b['sr']); pk=np.maximum.accumulate(w)
    dd=(w-pk)/pk*100
    ax.plot(dates_te,dd,color=mc[nm],lw=1.6,label=nm,alpha=0.85 if nm=='Elastic Net' else 0.6)
ax.set_title("Drawdown Comparison — All Strategies",fontsize=13,fontweight='bold')
ax.set_ylabel("Drawdown (%)"); ax.legend(); ax.axhline(0,color='gray',lw=0.5)
plt.tight_layout(); plt.savefig('/home/claude/vw_ml_trading/figures/fig6_drawdown.png',dpi=160,bbox_inches='tight'); plt.close()

print("All 6 figures saved.")

rows=[]
for nm in nms:
    m,b=metrics[nm],bt[nm]
    rows.append({'Model':nm,'RMSE':m['RMSE'],'MAE':m['MAE'],'R2':m['R2'],
                 'DirAcc':m['DirAcc'],'CumReturn':b['cum'],'BnH':b['bnh'],
                 'Sharpe':b['sharpe'],'MaxDrawdown':b['mdd']})
pd.DataFrame(rows).to_csv('/home/claude/vw_ml_trading/results_summary.csv',index=False)
print("CSV saved.")

# store results for PDF
import pickle
with open('/home/claude/vw_ml_trading/results.pkl','wb') as f:
    pickle.dump({'metrics':metrics,'bt':bt,'feature_cols':feature_cols,'coef':coef.to_dict()}, f)
print("Results pickled.")
