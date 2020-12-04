import os
import re
import argparse
import sys

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Running an experiment')
    parser.add_argument('--mode', dest='mode', help='mode to use',
                        default='test', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args
pattern = re.compile('(\d+)MiB / (\d+)MiB')
count = 0


if __name__ == '__main__':
    args = parse_args()
    while True:
        results = os.popen('nvidia-smi')
        res = results.read()
        mat = pattern.findall(res)
        flag = False
        count = 0
        for used, total in mat:
            used = int(used)
            if used < 4000:
                print('in')
                flag = True
                break
            count += 1
        if flag == True:
            ###run code####
            if args.mode == 'test':
                os.system('python ./test/cal_airplane_200_20.py --gpu %d' % count)
            elif args.mode == 'hold':
                os.system('python ./mvcnn/train_hold.py --gpu %d' % count)
            break


    ###run hold####
    #os.system('python ./mvcnn/train_hold.py --gpu %d'%count)