from __future__ import print_function
import numpy as np
import h5py

if __name__ == '__main__':
    print('Load val data')
    hf_val = h5py.File('data/val36.hdf5', 'r')
    print('Transfer val data')
    image_features_val36 = np.array(hf_val.get('image_features'))
    spatials_val36 = np.array(hf_val.get('spatial_features'))
    del hf_val
    print('Saving val data')
    np.save('data/image_features_val36.npy', image_features_val36)
    np.save('data/spatials_val36.npy', spatials_val36)
    del image_features_val36, spatials_val36

    print('Load Train data')
    hf_train = h5py.File('data/train36.hdf5', 'r')
    print('Transfer Train data')
    image_features_train36 = np.array(hf_train.get('image_features'))
    spatials_train36 = np.array(hf_train.get('spatial_features'))
    del hf_train
    print('Saving Train data')
    np.save('data/image_features_train36.npy', image_features_train36)
    np.save('data/spatials_train36.npy', spatials_train36)
    del image_features_train36, spatials_train36