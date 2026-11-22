"""Export the same full-held-out metrics shown in TensorBoard."""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args = parser.parse_args()
    folder = args.run / 'evaluation'
    rows = json.loads((folder/'checkpoint-curve.json').read_text())['rows']
    fig, axes = plt.subplots(2,2,figsize=(11,7),sharex=True)
    for column, interface in enumerate(('motion','human')):
        for index, metric in enumerate(('survival','mean_norm_return')):
            axis = axes[index,column]
            for robot in ('all','q1','bumi','g1'):
                selected = sorted((r for r in rows if r['interface']==interface and r['robot']==robot),key=lambda r:r['step'])
                axis.plot([r['step'] for r in selected],[r[metric] for r in selected],marker='o',
                    linewidth=2.5 if robot=='all' else 1.2,label=robot.upper())
            axis.grid(alpha=.25)
            axis.set_ylabel('Survival' if index==0 else 'Mean normalized return')
            if index==0:
                axis.set_title('Motion' if interface=='motion' else 'Mixed five-point')
                axis.set_ylim(0,1.02); axis.yaxis.set_major_formatter(PercentFormatter(1))
                axis.legend()
            else:
                axis.set_xlabel('Training step'); axis.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    fig.suptitle('Full held-out evaluation: 1,139 clips per robot, seed 0')
    fig.tight_layout()
    temporary = folder/'checkpoint-curve.tmp.png'
    fig.savefig(temporary,dpi=160)
    plt.close(fig)
    os.replace(temporary,folder/'checkpoint-curve.png')


if __name__=='__main__':
    main()
