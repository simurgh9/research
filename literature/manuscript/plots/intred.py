from matplotlib import pyplot as plt
# plt.rcParams['figure.dpi'] = 300

# files = [
#     "paper40_237413_212828.txt",
#     "paper50_576969_522364.txt",
#     "paper60_1221209_1113319.txt",
#     "paper70_2260926_2071820.txt",
#     "paper80_3643144_3351770.txt",
#     "paper90_6028468_5564035.txt",
#     "paper100_9012027_8339504.txt"
# ]

files = [
    "col40_1740.txt",
    "col50_1929.txt",
    "col60_2101.txt",
    "col70_2254.txt",
    "col80_2397.txt",
    # "col90_2544.txt",
    # "col100_2667.txt"
]

# lim = {40: 9, 50: 13, 60: 13, 70: 18, 80: 17, 90: 18, 100: 12}  # paper
lim = {40: 8, 50: 9, 60: 17, 70: 17, 80: 18, 90: -1, 100: -1}  # challenge
prefix = "../data/"

for name in files:
    bests, means = [], []
    path = prefix + name
    with open(path, 'r') as fl:
        lines = [line.strip() for line in fl.readlines() if line != '\n']
    GH = int(name.split('_')[-1][:-4])
    d = int(name.split('_')[0][len('col'):])  # change this when you change problem sets
    n = int(lines[0].split(':')[-1])

    i, seconds = 0, 0
    for line in lines[2:]:
        if i >= lim[d] or line.startswith('Total time: '):
            break;
        line = line.strip().replace(',', '').replace('s', '').split(' ')
        i = int(line[0])
        seconds += float(line[4])
        bests.append(float(line[1]))
        means.append(float(line[2]))

    title = f'Dimensiosn: {d}, '
    title += f'Population Size: {n}\n'
    title += f'Shortest Vector Length: {bests[-1]}\n'
    title += f'Seconds Taken: {seconds:.2f}s\n'
    title += f'Generations Taken: {lim[d] + 1}'
    plt.title(title)
    plt.xlabel('Generation Index')
    plt.ylabel('Euclidean Norm')
    plt.plot(bests, 'kD-', mfc='red', mec='k', alpha=0.5, label='Shortest Vector Length')
    plt.plot(means, 'ko-', mfc='green', mec='k', alpha=0.5, label='Mean Vector Length')
    plt.ticklabel_format(axis='both', style='plain')
    plt.tight_layout()
    plt.legend()
    # plt.savefig(f'../media/col{d}.png')  # change this when you change problem sets
    plt.show()
