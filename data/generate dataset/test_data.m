
clear;
clc;

addpath('./utils/');

benchmark = 'B100';

data_path = ['../test_data/' benchmark];
save_path = ['../test_data/mat/' benchmark];

scales = [2 3 4];


%% image list
image_list = [];
image_list = [image_list; dir(fullfile(data_path, '*.bmp'))];
image_list = [image_list; dir(fullfile(data_path, '*.jpg'))];

%% generate target directory

if ~exist(save_path, 'dir')
    mkdir(save_path);
end

%% generate test dataset

for i = 1:numel(image_list)
    file_info = image_list(i);
    disp([int2str(i) ' ' file_info.name]);
    file_path = fullfile(data_path, file_info.name);
    name = split(file_info.name, '.');
    
    ground_truth = imread(file_path);
    
    for s = 1:numel(scales)
        ground_truth_ = modcrop(ground_truth, scales(s));
        
        % for color images, keep a chrominance components for visualization
        if size(ground_truth_, 3) == 3
            
            ground_truth_ = rgb2ycbcr(ground_truth_);
            
            cb = ground_truth_(:, :, 2);
            cr = ground_truth_(:, :, 3);
            
            save(fullfile(save_path, [name{1} '_' int2str(scales(s)) '_cb']), 'cb');
            save(fullfile(save_path, [name{1} '_' int2str(scales(s)) '_cr']), 'cr');
        end
        
        gt = im2double(ground_truth_(:, :, 1));
        [h, w] = size(gt);
        lr = imresize(gt, 1/scales(s), 'bicubic');
        
        save(fullfile(save_path, [name{1} '_' int2str(scales(s)) '_gt']), 'gt');
        save(fullfile(save_path, [name{1} '_' int2str(scales(s)) '_lr']), 'lr');
        
        % disp(['file name : ', file_info.name, ' Scale = ', int2str(scales(s)), ' PSNR = ', num2str(psnr(lr, gt))]);
    end    
end