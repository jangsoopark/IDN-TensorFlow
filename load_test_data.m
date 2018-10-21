
function [hr, lr, chroma] = load_test_data(file_name, scale)
    image = imread(file_name);
    image = modcrop(image, scale);
    chroma = zeros(size(image, 1), size(image, 2), 2);
    if size(image, 3) == 3
        image = rgb2ycbcr(image);
        chroma = image(:, :, 2: 3);
        % chroma = im2double(chroma);
    end
    
    image = image(:, :, 1);
    image = im2double(image(:, :, 1));
    
    hr = image;
    [h, w] = size(hr);
    
    lr = imresize(imresize(hr, 1 / scale, 'bicubic'), [h, w], 'bicubic');    
end
