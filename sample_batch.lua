--[[
    Data transform functions.
]]

require 'torch'
require 'image'

paths.dofile('util/img.lua')


------------------------------------------------------------------------------------------------------------

--[[ Function for data augmentation, randomly samples on a normal distribution. ]]--
local function rnd(x)
    return math.max(-2*x, math.min(2*x, torch.randn(1)[1]*x))
end


-------------------------------------------------------------------------------
-- Transform data for one object sample
-------------------------------------------------------------------------------

function transform_data(img, params)
    assert(img)
    assert(params)

    -- Crop image
    local img_transf = crop2(img.img, img.center, img.scale, params.rotation, opt.inputRes)

    if not img_transf then return nil end

    -- Flipping
    if params.flip then
        img_transf = flip(img_transf)
    end

    -- color augmentation
    img_transf[1]:mul(params.color_jit[1]):clamp(0,1)
    img_transf[2]:mul(params.color_jit[2]):clamp(0,1)
    img_transf[3]:mul(params.color_jit[3]):clamp(0,1)

    -- output
    return img_transf
end


-------------------------------------------------------------------------------
-- Get a batch of data samples
-------------------------------------------------------------------------------

local function get_random_transforms(is_train)
    assert(is_train)

    local rot = 0
    local scale = 1
    local color_jit = {brightness = 1, contrast = 1, saturation = 1}
    local flip = false

    if is_train then
        -- scale
        scale = scale * (2 ^ rnd(opt.scale))

        -- rotation
        rot = rnd(opt.rotate)
        if torch.uniform() <= opt.rotRate then
            rot = 0
        end

        -- Flipping
        if torch.uniform() < .5 then
            flip = true
        end

        -- color augmentation
        color_jit = {torch.uniform(0.6, 1.4),
                     torch.uniform(0.6, 1.4),
                     torch.uniform(0.6, 1.4)}
    end

    return {
        rotation = rot,
        scale = scale,
        color_jit = color_jit,
        flipping = flip
    }
end

------------------------------------------------------------------------------------------------------------

--[[ Fetch data (images + label) from a single video ]]--
local function fetch_single_data(data_loader, idx, is_train)
    assert(data_loader)
    assert(idx)
    assert(is_train)

    local imgs, label = data_loader.loader(idx)

    if not imgs then
        return {}  -- return empty table to discard this image
    end

    -- select some random transformations to apply to the entire set of images
    local params_transform = get_random_transforms(is_train)

    -- apply transforms to all images
    local imgs_transf = {}
    for i=1, #imgs do
        local new_img = transform_data(imgs[i], params_transform)
        if not new_img then return {} end  -- skip this round of data/transforms if any error occurs
        table.insert(imgs_transf, new_img)
    end

    -- Resize images (vgg16)
    local iW, iH = torch.random(1, opt.inputRes - 224), torch.random(1, opt.inputRes - 224)
    local imgs_resized = {}
    for i=1, #imgs_transf do
        local new_img = imgs_transf[i]:clone()
        if torch.random() > 0.5 then
            new_img = image.scale(new_img, 224, 224)
        else
            new_img = random_crop(new_img, 224, 224, iW, iH)
        end

        -- convert bo bgr
        new_img = new_img:index(1, torch.LongTensor{3,2,1})  -- bgr

        -- rescale pixels
        new_img:mul(opt.params.pixel_scale)

        -- normalize pixels
        for i=1, 3 do
            if opt.params.mean then new_img[i]:add(-opt.params.mean[i]) end
            if opt.params.std then new_img[i]:div(opt.params.std[i]) end
        end

        table.insert(imgs_resized, new_img)
    end

    return {imgs_transf, imgs_resized, label}
end

------------------------------------------------------------------------------------------------------------

--[[ Create/build a batch of images + label ]]--
local function get_batch(data_loader, batchSize, is_train)
    assert(data_loader)
    assert(batchSize)
    assert(is_train)

    local size = data_loader.size
    local max_attempts = 30
    local batchData, idxUsed = {}, {}

    for i=1, batchSize do
        local data = {}
        local attempts = 0
        while not next(data) do
            local idx = torch.random(1, size)
            if not idxUsed[idx] then
                data = fetch_single_data(data_loader, idx, is_train)
                idxUsed[idx] = 1

                -- increment attempts counter. This avoids infinite loops
                -- if it cannot find a valid image + annotations
                attempts = attempts + 1

                if attempts > max_attempts then
                    error('Reached the maximum number of attempts to find an unique batch: ' .. max_attempts)
                end
            end
        end
        table.insert(batchData, data)
    end
    return batchData
end

------------------------------------------------------------------------------------------------------------

function getSampleBatch(data_loader, batchSize, is_train)
    assert(data_loader)
    assert(batchSize)
    assert(is_train)

    -- get batch data
    local sample = get_batch(data_loader, batchSize, is_train)

    -- images (for body joints)
    local imgs_kps = torch.FloatTensor(batchSize, opt.seq_length,
                                          3, opt.inputRes, opt.inputRes):fill(0)
    for i=1, batchSize do
        for j=1, opt.seq_length do
            imgs_kps[i][j]:copy(sample[i][1][j])
        end
    end

    -- images (for body joints)
    local imgs_feats = torch.FloatTensor(batchSize, opt.seq_length,
                                          3, 224, 224):fill(0)
    for i=1, batchSize do
        for j=1, opt.seq_length do
            imgs_feats[i][j]:copy(sample[i][2][j])
        end
    end

    -- labels
    local labels_tensor = torch.IntTensor(batchSize, opt.seq_length):fill(0)
    for i=1, batchSize do
        labels_tensor[i]:fill(sample[i][3])
    end

    collectgarbage()

    return imgs_kps, imgs_feats, labels_tensor
end
