"""Plot saved measured outcomes; no hard-coded benchmark scores."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'benchmarks/policy-v1/primary/report.json').read_text())
control=json.loads((ROOT/'benchmarks/policy-v1/untrained-greedy-control.json').read_text())
from sys2.benchmarks import wilson

sizes=list(report['by_size'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'text.color':'#e8eef8','axes.labelcolor':'#c3d0e0','xtick.color':'#c3d0e0','ytick.color':'#c3d0e0'})
fig,ax=plt.subplots(figsize=(12,6.5),facecolor='#0b1120')
ax.set_facecolor('#0b1120')
series=[('Untrained policy, one attempt','#6f819e'),('Trained policy, one attempt','#55b9ff'),('Trained policy + backtracking','#5be2b1')]
positions=np.arange(len(sizes));width=.24
for index,(name,color) in enumerate(series):
    rates=[];errors=[]
    for size in sizes:
        summary=report['by_size'][size]
        rate=(control['summary'][size]['untrained_greedy_success'] if index==0 else summary['greedy' if index==1 else 'guided_search']['success_rate'])
        n=summary['greedy']['count'];lo,hi=wilson(round(rate*n),n)
        rates.append(rate*100);errors.append([max(0,rate-lo)*100,max(0,hi-rate)*100])
    x=positions+(index-1)*width
    ax.bar(x,rates,width,color=color,label=name,zorder=3)
    ax.errorbar(x,rates,yerr=np.array(errors).T,fmt='none',ecolor='#e1e8f5',capsize=3,linewidth=1,zorder=4)
    for xx,rate in zip(x,rates):ax.text(xx,rate+3,f'{rate:.1f}%',ha='center',fontsize=10,color=color)
ax.set_xticks(positions,[f'{size} × {size}' for size in sizes]);ax.set_ylim(0,116)
ax.set_ylabel('Verified complete routes (%)');ax.set_xlabel('Maze size · trained on 3×3 and 4×4 only',labelpad=14)
ax.set_yticks(range(0,101,20));ax.grid(axis='y',color='#27334b',alpha=.8,zorder=0)
for side in ['top','right']:ax.spines[side].set_visible(False)
for side in ['left','bottom']:ax.spines[side].set_color('#3d4c66')
ax.legend(loc='upper left',bbox_to_anchor=(0,1.19),ncol=3,frameon=False,fontsize=10,labelcolor='#c3d0e0')
fig.suptitle('Held-out maze solving · 32.10 MB scratch-trained policy',fontsize=18,x=.08,ha='left',y=.98)
fig.text(.08,.02,'300 fresh tasks per size · 95% Wilson intervals · training wall-layout/goal combinations excluded across all starts\nBacktracking supplies search completeness. Untrained controls use the same legal-move and no-revisit constraints.',fontsize=10,color='#9dacbf')
fig.subplots_adjust(left=.08,right=.98,bottom=.2,top=.79)
fig.savefig(ROOT/'evidence/benchmark-summary.png',dpi=160,facecolor=fig.get_facecolor())
fig.savefig(ROOT/'evidence/benchmark-summary.svg',facecolor=fig.get_facecolor())
print('Saved benchmark summary chart from measured records')
