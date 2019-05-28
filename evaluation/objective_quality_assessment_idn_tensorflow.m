clear;
clc;

addpath('./utils/');
benchmark_root = '../data/test_data/mat';
result_root = '../result/';

benchmark = 'Urban100';

scale = [2 3 4];

for s = 1:numel(scale)
    avg_psnr = 0;
    avg_ssim = 0;
    gt_list = dir(fullfile(benchmark_root, benchmark, ['*_', int2str(scale(s)), '_gt.mat']));
    sr_list = dir(fullfile(result_root, benchmark, int2str(scale(s)), '*.mat'));
    

    for i = 1:numel(gt_list)
        gt_path = gt_list(i).name;
        sr_path = sr_list(i).name;
    
        load(fullfile(benchmark_root, benchmark, gt_path));
        load(fullfile(result_root, benchmark, int2str(scale(s)), sr_path));
        sr = double(sr);
        
        sr = shave(sr, [scale(s), scale(s)]);
        gt = shave(gt, [scale(s), scale(s)]);
        avg_psnr = avg_psnr + psnr(sr, gt);
        avg_ssim = avg_ssim + ssim(sr, gt);
    end
    avg_psnr = avg_psnr / numel(gt_list);
    avg_ssim = avg_ssim / numel(gt_list);
    
    disp(['Scale : ', int2str(scale(s)), ', PSNR / SSIM = ', num2str(avg_psnr) , ' / ', num2str(avg_ssim)]);
end
