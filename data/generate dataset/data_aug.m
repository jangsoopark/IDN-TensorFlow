clear;
clc;

addpath('./utils/');

%% data path
data_path = '../train_data/291';
save_path = '../train_data/291-aug';

%% augmentation parameter
rotation_ = 0: 90: 270;
downscale_ = 1.0: -0.1: 0.6;
flip_ = 3: -2: 1;

%% generate target directory
if ~exist(save_path, 'dir')
    mkdir(save_path);
end

%% file path
file_list = [];
file_list = [file_list; dir(fullfile(data_path, '*.jpg'))];
file_list = [file_list; dir(fullfile(data_path, '*.bmp'))];

%% data augmentation
for i = 1:numel(file_list)
    
    disp(file_list(i).name);
    [add, im_name, type] = fileparts(file_list(i).name);
    
    %% read image
    image = imread(fullfile(data_path, file_list(i).name));
    image = rgb2ycbcr(image);
    image = im2double(image(:, :, 1));
    
    count = 0;
    %% rotation
    for r = 1:numel(rotation_)
        %% flip
        for f = 1:numel(flip_)
            %% down scale
            for s = 1:numel(downscale_)
                image_aug = image_augmentation(image, rotation_(r), flip_(f), downscale_(s));
                imwrite(image_aug, [fullfile(save_path, im_name) '_' num2str(count) '.bmp']);
                
                count = count + 1;
            end
        end
    end
end