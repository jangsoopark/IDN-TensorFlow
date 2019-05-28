clear;
clc;

benchmark_root = '../data/test_data/mat';
result_root = '../result/';

benchmark = 'Urban100';


scale = [2 3 4];

for s = 1:numel(scale)
    visualization_path = fullfile(benchmark, int2str(scale(s)));
    if ~exist(fullfile(benchmark, int2str(scale(s))), 'dir')
        mkdir(visualization_path);
    end

    gt_list = dir(fullfile(benchmark_root, benchmark, ['*_', int2str(scale(s)), '_gt.mat']));
    lr_list = dir(fullfile(benchmark_root, benchmark, ['*_', int2str(scale(s)), '_lr.mat']));
    sr_list = dir(fullfile(result_root, benchmark, int2str(scale(s)), '*.mat'));

    cb_list = dir(fullfile(benchmark_root, benchmark, ['*_', int2str(scale(s)), '_cb.mat']));
    cr_list = dir(fullfile(benchmark_root, benchmark, ['*_', int2str(scale(s)), '_cr.mat']));

    for i = 1:numel(gt_list)
        gt_path = gt_list(i).name;
        lr_path = lr_list(i).name;
        sr_path = sr_list(i).name;
        cb_path = cb_list(i).name;
        cr_path = cr_list(i).name;

        load(fullfile(benchmark_root, benchmark, gt_path));
        load(fullfile(benchmark_root, benchmark, lr_path));
        load(fullfile(result_root, benchmark, int2str(scale(s)), sr_path));
        load(fullfile(benchmark_root, benchmark, cb_path));
        load(fullfile(benchmark_root, benchmark, cr_path));

        % gt_ = uint8(gt * 255);
        % gt_color = ycbcr2rgb(cat(3, gt_, cb, cr));

        % lr_ = uint8(lr * 255);
        % lr_color = ycbcr2rgb(cat(3, lr_, cb, cr));

        sr_ = uint8(sr * 255);
        sr_color = ycbcr2rgb(cat(3, sr_, cb, cr));

        image_name = split(sr_list(i).name, '.');
        image_name = image_name{1};

        % imwrite(gt_color, fullfile(visualization_path, [image_name, '_gt.png']));
        % imwrite(lr_color, fullfile(visualization_path, [image_name, '_lr.png']));
        imwrite(sr_color, fullfile(visualization_path, [image_name, '_sr.png']));
    end
end