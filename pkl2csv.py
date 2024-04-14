import os
import pickle
from natsort import natsorted
from tqdm import tqdm
from glob import glob
import pandas

if __name__ == '__main__':

    os.makedirs('merged_data_pkl', exist_ok=True)

    X_data_files  = natsorted(glob('./train_pkl/train_data_person_*_data.pkl'))
    Y_label_files = natsorted(glob('./train_pkl/train_data_person_*_label.pkl'))

    for X_data_path, Y_label_path in zip(X_data_files, Y_label_files):
        filename = os.path.splitext(os.path.basename(X_data_path))[0].split('_')[-2]
        print(filename)
        with open(X_data_path, 'rb') as file:
            X_data = pickle.load(file)
        with open(Y_label_path, 'rb') as file:
            Y_label = pickle.load(file)
        
        print(X_data.shape, Y_label.shape)

        skip_window_size = 32
        label_cnt = 0

        for idx in tqdm(range(0, X_data.shape[0], skip_window_size)):
            data_frame = X_data[idx:idx+skip_window_size, :]
            
            skip_frame_size = 225

            for frame_idx in range(0, data_frame.shape[-1], skip_frame_size):
                if frame_idx!=0:
                    data_chunk = data_frame[:, frame_idx-25:frame_idx+skip_frame_size]
                else:
                    data_chunk = data_frame[:, frame_idx:frame_idx+skip_frame_size+25]
                save_path = './merged_data_pkl/{}_{}_{}.pkl'.format(filename, idx, frame_idx)
                with open(save_path, 'wb') as file:
                    pickle.dump({'data':data_chunk, 'label': Y_label[label_cnt]}, file)
            
            label_cnt += 1