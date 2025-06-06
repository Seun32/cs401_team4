import Augmentor
import os
import argparse

def makedir(path):
    '''
    if path does not exist in the file system, create it
    '''
    if not os.path.exists(path):
        os.makedirs(path)


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str)
parser.add_argument('--augment_times', type=int, default=10)
args = parser.parse_args()

augment_times = args.augment_times
# datasets_root_dir = '/content/drive/MyDrive/datasets/cub200_cropped/'
datasets_root_dir = args.data_path
if 'crop' in args.data_path:
    push_dir, train_dir = 'train_cropped/', 'train_cropped_augmented/'
else:
    push_dir, train_dir = 'train/', 'train_augmented/'

dir = os.path.join(datasets_root_dir, push_dir)
target_dir = os.path.join(datasets_root_dir, train_dir)

makedir(target_dir)
folders = [os.path.join(dir, folder) for folder in next(os.walk(dir))[1]]
target_folders = [os.path.join(target_dir, folder) for folder in next(os.walk(dir))[1]]

for i in range(len(folders)):
    fd = folders[i]
    tfd = target_folders[i]
    # rotation
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.rotate(probability=1, max_left_rotation=14, max_right_rotation=14)
    p.flip_left_right(probability=0.5)
    for i in range(augment_times):
        p.process()
    del p
    # skew
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.skew(probability=1, magnitude=0.2)  # max 45 degrees
    p.flip_left_right(probability=0.5)
    for i in range(augment_times):
        p.process()
    del p
    # shear
    p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    p.shear(probability=1, max_shear_left=10, max_shear_right=10)
    p.flip_left_right(probability=0.5)
    for i in range(augment_times):
        p.process()
    del p
    # random_distortion
    #p = Augmentor.Pipeline(source_directory=fd, output_directory=tfd)
    #p.random_distortion(probability=1.0, grid_width=10, grid_height=10, magnitude=5)
    #p.flip_left_right(probability=0.5)
    #for i in range(10):
    #    p.process()
    #del p