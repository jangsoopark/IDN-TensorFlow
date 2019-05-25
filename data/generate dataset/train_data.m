clear;
clc;

addpath('./utils/');

%% data path
data_path = '../train_data/291-aug';
save_path = '../train_data/idn_fine_tuning_x4.h5'; %idn_train_x4.h5';

%% data configuration
% scale | training | fine-tuning
%   2   | 29 / 58  |  39 / 78
%   3   | 15 / 45  |  26 / 78
%   4   | 11 / 44  |  19 / 76

lr_patch_size = 19;
gt_patch_size = 76;

padding = abs(lr_patch_size - gt_patch_size) / 2;

scale = 4;

stride_lr = lr_patch_size;
stride_gt = stride_lr * scale;

%% file path
file_list = [];
file_list = [file_list; dir(fullfile(data_path, '*.jpg'))];
file_list = [file_list; dir(fullfile(data_path, '*.bmp'))];


%% data pairs
lr_list = zeros(lr_patch_size, lr_patch_size, 1, 1);
gt_list= zeros(gt_patch_size, gt_patch_size, 1, 1);

count_gt = 0;
count_lr = 0;

%% generate dataset
for i = 1:numel(file_list)
    %% read image
    disp(file_list(i).name);
    image = imread(fullfile(data_path, file_list(i).name));
    image = im2double(image(:, :, 1));

    gt = modcrop(image, scale);
    [h, w] = size(gt);
    lr = imresize(gt, 1/scale, 'bicubic');

    %% ground truth
    for y = 1:stride_gt:h - gt_patch_size + 1
        for x = 1:stride_gt:w - gt_patch_size + 1
            gt_patch = gt(y: y + gt_patch_size - 1, x: x + gt_patch_size - 1);
            count_gt = count_gt + 1;

            gt_list(:, :, 1, count_gt) = gt_patch;
        end
    end

    [h, w] = size(lr);
    %% low resolution input
    for y = 1:stride_lr:h - lr_patch_size + 1
        for x = 1:stride_lr:w - lr_patch_size + 1
            lr_patch = lr(y: y + lr_patch_size - 1, x: x + lr_patch_size - 1);
            count_lr = count_lr + 1;

            lr_list(:, :, 1, count_lr) = lr_patch;
        end
    end


    
end

order = randperm(count_gt);
lr_list = lr_list(:, :, 1, order);
gt_list = gt_list(:, :, 1, order); 

%% writing to HDF5
chunksz = 64;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count_gt/chunksz)
    last_read=(batchno-1)*chunksz;
    batch_lr = lr_list(:,:,1,last_read+1:last_read+chunksz); 
    batch_gt = gt_list(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('lr',[1,1,1,totalct+1], 'gt', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(save_path, batch_lr, batch_gt, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(save_path);