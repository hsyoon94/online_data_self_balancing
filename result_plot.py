import os
import argparse

from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas
import numpy


#python plot.py --exps HalfCheetah_q3_action128 HalfCheetah_q3_action4096 \
#HalfCheetah_q3_action16384 --save HalfCheetah_q3_actions

#python result_plot.py --exps DSS DSS-DBC DSS-DF DSS-DE --save True

parser = argparse.ArgumentParser()
parser.add_argument('--exps',  nargs='+', type=str)
parser.add_argument('--save', type=str, default=None)
parser.add_argument('--env-name', default='MountainOldCarContinuous-v1',
                    help='environment to train on (default: PongNoFrameskip-v4)')

args = parser.parse_args()


def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.2fM' % (x*1e-6)


def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fK' % (x*1e-3)

def normals(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x)

font = {'family' : 'Dejavu Sans',
        'size'   : 24}

plt.rc('font', **font)

# formatter = FuncFormatter(thousands)
formatter = FuncFormatter(normals)

f, ax = plt.subplots(1, 1, figsize=(17, 8))
ax.xaxis.set_major_formatter(formatter)

ax.patch.set_facecolor('lavender')
ax.set_facecolor((234/256, 234/256, 243/256))
# ax.patch.set_alpha(0.5)

for i, exp in enumerate(args.exps):
    log_fname = os.path.join('log', '210217', '1104', exp + '.csv')
    csv = pandas.read_csv(log_fname)

    # color = cm.viridis(i / float(len(args.exps)))
    # colors = ['crimson', 'orchid', 'orange','teal']
    colors = [ 'orange', 'magenta', 'purple', 'olivedrab', 'steelblue', 'peru', 'firebrick', 'darkolivegreen', 'darkblue', 'teal', 'coral', 'lightblue', 'lime', 'orange',
              'darkgreen', 'tan', 'salmon', 'gold', 'turquoise',
              'blue', 'green', 'red', 'magenta', 'yellow', 'black', 'pink',]

    ax.plot(csv['Timesteps'], csv['ReturnAvg'], color=colors[i], label=exp, linewidth='2.0')
    ax.fill_between(csv['Timesteps'], csv['ReturnAvg'] - csv['ReturnStd'], csv['ReturnAvg'] + csv['ReturnStd'], color=colors[i], alpha=0.2)

ax.legend()
# For loss
ax.set_xlabel('Number of training days', fontsize=24)
ax.set_ylabel('Average loss', fontsize=24)

# For entropy
# ax.set_xlabel('Number of training epochs', fontsize=24)
# ax.set_ylabel('Normalized entropy', fontsize=24)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax.margins(0.0, 0.05)
ax.set_axisbelow(True)

ax.grid(linestyle='-', linewidth='1.0', color='white')

if args.save:
    os.makedirs('plots', exist_ok=True)
    # f.savefig(os.path.join('plots', args.save + '.jpg'))
    f.savefig(os.path.join('log', '210217', '1104', exp + '.pdf'))
else:
    plt.show()