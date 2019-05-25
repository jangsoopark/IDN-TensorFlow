
clear;
clc;

addpath('utils');

%% benchmark image
benchmark = 'Urban100';
scale = 2;
image_name = 'img002';
ext = '.png';

gt_path = fullfile(benchmark, int2str(scale), [image_name, '_gt', ext]);
lr_path = fullfile(benchmark, int2str(scale), [image_name, '_lr', ext]);
sr_path = fullfile(benchmark, int2str(scale), [image_name, '_sr', ext]);

%% read image
gt = imread(gt_path);
lr = imread(lr_path);
sr = imread(sr_path);

imshow(gt);
[x, y] = ginput(2);

pos = rect_pos(x, y);
rectangle('Position', pos, 'EdgeColor', 'r', 'LineWidth', 5);

figure, imshow(lr);
rectangle('Position', pos, 'EdgeColor', 'r', 'LineWidth', 5);

figure, imshow(sr);
rectangle('Position', pos, 'EdgeColor', 'r', 'LineWidth', 5);

figure, imshow(gt(y(1): y(2), x(1): x(2), :));
figure, imshow(lr(y(1): y(2), x(1): x(2), :));
figure, imshow(sr(y(1): y(2), x(1): x(2), :));
