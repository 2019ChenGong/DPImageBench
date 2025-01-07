import os
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument('--D_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--D_num', type=int, default=1000)
args = parser.parse_args()

gfile_stream = open(args.output_path, 'a')

finished_num = 0
while True:
    finished_num_cur = 0
    for netD_id in range(args.D_num):
        save_subdir = os.path.join(args.D_path, 'netD_%d' % netD_id)
        if os.path.exists(os.path.join(save_subdir, 'netD.pth')):
            finished_num_cur += 1
    if finished_num_cur > finished_num:
        finished_num = finished_num_cur
        gfile_stream.write("{} / {} discriminators have been pre-trained\n".format(finished_num, args.D_num))
    if finished_num == args.D_num:
        break
    else:
        time.sleep(300)