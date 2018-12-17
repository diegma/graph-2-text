import os, argparse, sys
import sr11_onmtgcn_input

def prepare_files_ter(refsFile, predsFile):
    """
    Generate files for TER input.
    :param refsFile: full path to gold references file
    :param predsFile: full path to predictions file
    :return:
    """
    references = []  # each element is a list of references
    # complete refs with references for all sents
    with open(refsFile, 'r') as f:
        for i, line in enumerate(f):
            references.append([line.strip() + ' (id' + str(i) + ')\n'])

    outdir = refsFile.split(os.path.basename(refsFile))[0]

    # create a file with only one reference for TER
    with open(outdir + 'sr-references-ter.txt', 'w+') as f:
        for ref in references:
            f.write(''.join(ref))

    # prepare generated hypotheses
    with open(predsFile, 'r') as f:
        geners = [line.strip() + ' (id' + str(i) + ')\n' for i, line in enumerate(f)]
    with open(predsFile.replace('.txt','-ter.txt'), 'w+') as f:
        f.write(''.join(geners))

    print('SR files for TER generated successfully.')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pred', help="Path to predictions file.", required=True)
    parser.add_argument('--gold', help="Path to gold references file.", required=True)

    args = parser.parse_args(sys.argv[1:])

    #topdir = './'
    #prepare_files_ter(topdir, args)
    prepare_files_ter(args.gold, args.pred)