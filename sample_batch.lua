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
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')

    local rot = 0
    local scale = 1
    local color_jit = {1, 1, 1}
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

local function get_center_crop_bbox(img)
    local width, height = img:size(3), img:size(2)
    local pad = math.abs((width - height)/2)
    if width >= height then
        return torch.FloatTensor({pad, 1, pad+height -1, height})
    else
        return torch.FloatTensor({1, pad, width, pad+width -1})
    end
end

------------------------------------------------------------------------------------------------------------

local function fetch_subset_images(imgs, seq_length)
    local out_imgs = {}
    local idx_ini = math.random(1, #imgs - opt.seq_length) - 1  -- select a subset of images from the video
    for i=1, seq_length do
        table.insert(out_imgs, imgs[idx_ini + i])
    end
    return out_imgs
end

------------------------------------------------------------------------------------------------------------

--[[ Fetch data (images + label) from a single video ]]--
local function fetch_single_data(data_loader, idx, is_train, use_subset)
    assert(data_loader)
    assert(idx)
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')
    assert(use_subset ~= nil)

    -- get images + label from a video
    local imgs, label = data_loader.loader(idx)

    -- select a subset of images from the video
    if use_subset then
        imgs = fetch_subset_images(imgs, opt.seq_length)
    end

    -- select some random transformations to apply to the entire set of images
    local params_transform = get_random_transforms(is_train)


    local imgs_transf = {}
    local prev_center, prev_scale = imgs[1].center, imgs[1].scale
    local prev_bbox = imgs[1].bbox
    if prev_bbox:sum() == 0 or prev_bbox[4]-prev_bbox[2] < 20 then
        prev_bbox = get_center_crop_bbox(imgs[1].img)
        prev_center = torch.FloatTensor({(prev_bbox[1] + prev_bbox[3])/2,
                                         (prev_bbox[2] + prev_bbox[4])/2})
        prev_scale = (prev_bbox[4]-prev_bbox[2]) / 200 * 1.5
    end
    for i=1, #imgs do
        local img = imgs[i].img
        local scale = imgs[i].scale
        local center = imgs[i].center
        local bbox = imgs[i].bbox

        if scale < 0 or bbox:sum() == 0 or bbox[4]-bbox[2] < 20 then
            imgs[i].bbox = prev_bbox
            imgs[i].center = prev_center
            imgs[i].scale = prev_scale
        end

        local new_img = transform_data(imgs[i], params_transform)
        if not new_img then return {} end  -- skip this round of data/transforms if any error occurs
        table.insert(imgs_transf, new_img)

        if not (scale < 0 or bbox:sum() == 0 or bbox[4]-bbox[2] < 20) then
            prev_bbox = bbox
            prev_center = center
            prev_scale = scale
        end
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
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')

    local size = data_loader.size
    local max_attempts = 30
    local batchData, idxUsed = {}, {}

    for i=1, batchSize do
        local data = {}
        local attempts = 0
        while not next(data) do
            local idx = torch.random(1, size)
            if not idxUsed[idx] then
                data = fetch_single_data(data_loader, idx, is_train, true)
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

--[[ Returns a batch of image sequences (used for training). ]]--
function getSampleBatch(data_loader, batchSize, is_train)
    assert(data_loader)
    assert(batchSize)
    assert(is_train ~= nil)
    assert(type(is_train) == 'boolean')

    -- get batch data
    local sample = get_batch(data_loader, batchSize, is_train)

    -- images (for body joints)
    local imgs_kps = torch.FloatTensor(batchSize, opt.seq_length,
                                       3, opt.inputRes, opt.inputRes):fill(0)
    for i=1, batchSize do
        local prev_sample = sample[i][1][1]
        for j=1, opt.seq_length do
            local sample_ = sample[i][1][j]
            if sample_ then
                prev_sample = sample_
            else
                sample_ = prev_sample
            end
            imgs_kps[i][j]:copy(sample_)
        end
    end

    -- images (for body joints)
    local imgs_feats = torch.FloatTensor(batchSize, opt.seq_length,
                                         3, 224, 224):fill(0)
    for i=1, batchSize do
        local prev_sample = sample[i][2][1]
        for j=1, opt.seq_length do
            local sample_ = sample[i][2][j]
            if sample_ then
                prev_sample = sample_
            else
                sample_ = prev_sample
            end
            imgs_feats[i][j]:copy(sample_)
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


------------------------------------------------------------------------------------------------------------

--[[ Returns a full sequence of images from a video (used for testing). ]]--
function getSampleTest(data_loader, idx)
    assert(data_loader)

    -- get batch data
    local sample = fetch_single_data(data_loader, idx, false, false)
    local seq_length = #sample[1]

    -- images (for body joints)
    local imgs_kps = torch.FloatTensor(1, seq_length, 3, opt.inputRes, opt.inputRes):fill(0)
    for j=1, seq_length do
        imgs_kps[1][j]:copy(sample[1][j])
    end

    -- images (for body joints)
    local imgs_feats = torch.FloatTensor(1, seq_length, 3, 224, 224):fill(0)
    for j=1, seq_length do
        imgs_feats[1][j]:copy(sample[2][j])
    end

    -- labels
    local labels_tensor = torch.IntTensor(1, seq_length):fill(sample[3])

    collectgarbage()

    return imgs_kps, imgs_feats, labels_tensor
end