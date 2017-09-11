--[[
    Image coordinates/cropping transformation functions.
]]


require 'image'


-------------------------------------------------------------------------------
-- Coordinate transformation
-------------------------------------------------------------------------------

function getTransform(center, scale, rot, res)
    local h = 200 * scale
    local t = torch.eye(3)

    -- Scaling
    t[1][1] = res / h
    t[2][2] = res / h

    -- Translation
    t[1][3] = res * (-center[1] / h + .5)
    t[2][3] = res * (-center[2] / h + .5)

    -- Rotation
    if rot ~= 0 then
        rot = -rot
        local r = torch.eye(3)
        local ang = rot * math.pi / 180
        local s = math.sin(ang)
        local c = math.cos(ang)
        r[1][1] = c
        r[1][2] = -s
        r[2][1] = s
        r[2][2] = c
        -- Need to make sure rotation is around center
        local t_ = torch.eye(3)
        t_[1][3] = -res/2
        t_[2][3] = -res/2
        local t_inv = torch.eye(3)
        t_inv[1][3] = res/2
        t_inv[2][3] = res/2
        t = t_inv * r * t_ * t
    end

    return t
end


function transform(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1],pt_[2] = pt[1]-1,pt[2]-1

    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2)

    return new_point:int():add(1)
end


function transformBenchmark(pt, center, scale, rot, res, invert)
    local pt_ = torch.ones(3)
    pt_[1] = pt[1]
    pt_[2] = pt[2]
    local t = getTransform(center, scale, rot, res)
    if invert then
        t = torch.inverse(t)
    end
    local new_point = (t*pt_):sub(1,2):int()
    return new_point
end

-------------------------------------------------------------------------------
-- Cropping
-------------------------------------------------------------------------------

function checkDims(dims)
    return dims[3] < dims[4] and dims[5] < dims[6]
end


function crop2(img, center, scale, rot, res)
    local ndim = img:nDimension()
    if ndim == 2 then img = img:view(1,img:size(1),img:size(2)) end
    local ht,wd = img:size(2), img:size(3)
    local tmpImg,newImg = img, torch.zeros(img:size(1), res, res)

    -- Modify crop approach depending on whether we zoom in/out
    -- This is for efficiency in extreme scaling cases
    local scaleFactor = (200 * scale) / res
    if scaleFactor < 2 then
        scaleFactor = 1
    else
        local newSize = math.floor(math.max(ht,wd) / scaleFactor)
        if newSize < 2 then
            -- Zoomed out so much that the image is now a single pixel or less
            if ndim == 2 then
                newImg = newImg:view(newImg:size(2),newImg:size(3))
            end
            return newImg
        else
           tmpImg = image.scale(img,newSize)
           ht,wd = tmpImg:size(2),tmpImg:size(3)
        end
    end

    -- Calculate upper left and bottom right coordinates defining crop region
    local c,s = center:float()/scaleFactor, scale/scaleFactor
    local ul = transform({1,1}, c, s, 0, res, true)
    local br = transform({res+1,res+1}, c, s, 0, res, true)
    if scaleFactor >= 2 then br:add(-(br - ul - res)) end

    -- If the image is to be rotated, pad the cropped area
    local pad = math.ceil(torch.norm((ul - br):float())/2 - (br[1]-ul[1])/2)
    if rot ~= 0 then ul:add(-pad); br:add(pad) end

    -- Define the range of pixels to take from the old image
    local old_ = {1,-1,math.max(1, ul[2]), math.min(br[2], ht+1) - 1,
                       math.max(1, ul[1]), math.min(br[1], wd+1) - 1}
    -- And where to put them in the new image
    local new_ = {1,-1,math.max(1, -ul[2] + 2), math.min(br[2], ht+1) - ul[2],
                       math.max(1, -ul[1] + 2), math.min(br[1], wd+1) - ul[1]}

    -- Initialize new image and copy pixels over
    local newImg = torch.zeros(img:size(1), br[2] - ul[2], br[1] - ul[1])
    if not pcall(function() newImg:sub(unpack(new_)):copy(tmpImg:sub(unpack(old_))) end) then
       print("Error occurred during crop!")
       return nil
    end

    if rot ~= 0 then
        -- Rotate the image and remove padded area
        newImg = image.rotate(newImg, rot * math.pi / 180, 'bilinear')
        newImg = newImg:sub(1,-1,pad+1,newImg:size(2)-pad,pad+1,newImg:size(3)-pad):clone()
    end

    if scaleFactor < 2 then newImg = image.scale(newImg,res,res) end
    if ndim == 2 then newImg = newImg:view(newImg:size(2),newImg:size(3)) end
    return newImg
end

------------------------------------------------------------------------------------------------------------

function random_crop(img, width, height, iW, iH)
    return img[{{}, {iW, iW+width -1}, {iH, iH + height -1}}]
end

------------------------------------------------------------------------------------------------------------

function resize_image(input, size, interpolation)
    local interpolation = interpolation or 'bicubic'
    local w, h = input:size(3), input:size(2)
    if (w <= h and w == size) or (h <= w and h == size) then
        return input
    end
    if w < h then
        return image.scale(input, size, h/w * size, interpolation)
    else
        return image.scale(input, w/h * size, size, interpolation)
    end
end


-------------------------------------------------------------------------------
-- Flipping functions
-------------------------------------------------------------------------------

function flip(x)
    local y = torch.FloatTensor(x:size())
    for i = 1, x:size(1) do
        image.hflip(y[i], x[i]:float())
    end
    return y:typeAs(x)
end