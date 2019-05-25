function image = modcrop(image, modulo)
if size(image,3) == 1
    sz = size(image);
    sz = sz - mod(sz, modulo);
    image = image(1:sz(1), 1:sz(2));
else
    tmpsz = size(image);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    image = image(1:sz(1), 1:sz(2),:);
end

