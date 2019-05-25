
function image_aug = image_augmentation(image, r, f, d)

image_aug = imresize(image, d, 'bicubic');
image_aug = imrotate(image_aug, r);

if f == 1 || f == 2
    image_aug = flip(image_aug, f);
end

end

