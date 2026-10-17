"""Shared journal-style plotting helpers (8.1 in wide, print-safe typography)."""
from pathlib import Path
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

NAVY, TEAL, BLUE = '#193A56', '#168980', '#5688AD'
OCHRE, CORAL, PURPLE = '#C7923A', '#B85B61', '#81749E'
INK, MUTED, GRID = '#263D4E', '#72818D', '#E3E9EE'
PALETTE = [NAVY, TEAL, OCHRE, BLUE, CORAL, PURPLE]
SEQUENTIAL = LinearSegmentedColormap.from_list('academic_teal', ['#F3F7F8', '#D8EAE6', '#8FC5BE', TEAL, NAVY])

def setup():
    for p in ['C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/msyhbd.ttc']:
        if Path(p).exists(): font_manager.fontManager.addfont(p)
    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Arial','DejaVu Sans','Microsoft YaHei','Noto Sans CJK SC'],
        'font.size':11.5, 'axes.labelsize':11.5, 'axes.titlesize':11.5, 'xtick.labelsize':10.7, 'ytick.labelsize':10.7,
        'legend.fontsize':10.5, 'text.color':INK, 'axes.labelcolor':INK, 'axes.edgecolor':GRID,
        'xtick.color':MUTED, 'ytick.color':MUTED, 'grid.color':GRID, 'grid.linewidth':.65, 'axes.axisbelow':True,
        'axes.spines.top':False, 'axes.spines.right':False, 'axes.spines.left':False,
        'xtick.major.size':0, 'ytick.major.size':0, 'pdf.fonttype':42, 'ps.fonttype':42, 'svg.fonttype':'path',
        'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

def heading(fig, title, subtitle, x=.075):
    fig.text(x,.966,title,ha='left',va='top',fontsize=14,weight='bold',color=NAVY)
    fig.text(x,.905,textwrap.fill(subtitle,width=95),ha='left',va='top',fontsize=10.6,color=MUTED,linespacing=1.35)

def canvas(n=1, height=4.25, widths=None, title='', subtitle='', left=.09, right=.985, bottom=.18, top=.765, wspace=.48):
    fig, axes=plt.subplots(1,n,figsize=(8.1,height),squeeze=False,
        gridspec_kw={'width_ratios':widths} if widths else None)
    fig.subplots_adjust(left=left,right=right,bottom=bottom,top=top,wspace=wspace)
    for ax in axes[0]: ax.grid(axis='y')
    if title: heading(fig,title,subtitle)
    return fig,list(axes[0])

def panel(ax, label, title=''):
    width=max(16,int(ax.get_position().width*ax.figure.get_figwidth()*72/(11.5*.51)))
    wrapped=textwrap.fill(title,width=width-3) if title else ''
    ax.set_title(f'{label}   {wrapped}' if title else label,loc='left',pad=11,fontsize=11.5,weight='bold',color=NAVY)

def heatmap(ax, values, xlabels, ylabels, vmin=0, vmax=1, fmt='.2f', cmap=None, fontsize=10.8):
    arr=np.asarray(values,dtype=float)
    im=ax.imshow(arr,vmin=vmin,vmax=vmax,cmap=cmap or SEQUENTIAL,aspect='auto')
    ax.set_xticks(np.arange(len(xlabels)),xlabels);ax.set_yticks(np.arange(len(ylabels)),ylabels);ax.grid(False)
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            v=arr[i,j]
            ax.text(j,i,format(v,fmt),ha='center',va='center',fontsize=fontsize,
                color='white' if (v-vmin)/max(vmax-vmin,1e-12)>.60 else INK)
    ax.set_xticks(np.arange(-.5,arr.shape[1],1),minor=True);ax.set_yticks(np.arange(-.5,arr.shape[0],1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2);ax.tick_params(which='minor',length=0)
    for s in ax.spines.values():s.set_visible(False)
    return im

def colorbar(fig, im, ax, label, **kwargs):
    cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04,**kwargs);cb.set_label(label,fontsize=10.6);cb.ax.tick_params(labelsize=10.5);cb.outline.set_visible(False)
    return cb

def box_points(ax, groups, labels, colors=None, ylabel='', log=False, jitter=.065):
    colors=colors or PALETTE
    positions=np.arange(len(groups))
    bp=ax.boxplot(groups,positions=positions,widths=.38,patch_artist=True,showfliers=False,
        medianprops={'color':NAVY,'linewidth':1.4},boxprops={'linewidth':.9},whiskerprops={'linewidth':.9},capprops={'linewidth':.9})
    for i,p in enumerate(bp['boxes']):p.set_facecolor(colors[i]);p.set_alpha(.14);p.set_edgecolor(colors[i])
    for i,v in enumerate(groups):
        # Deterministic horizontal displacement; no synthetic measurement is introduced.
        offsets=np.linspace(-jitter,jitter,len(v)) if len(v)>1 else [0]
        ax.scatter(i+np.asarray(offsets),v,s=23,color=colors[i],alpha=.70,edgecolors='white',linewidths=.35,zorder=4)
    ax.set_xticks(positions,labels);ax.set_ylabel(ylabel)
    if log:ax.set_yscale('log')
    return bp
