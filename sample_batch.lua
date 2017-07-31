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

    -- select some random transformations to apply to the entire set of images
    local params_transform = get_random_transforms(is_train)

    -- apply transforms to all images
    local imgs_transf = {}
    for i=1, #imgs do
        local new_img = transform_data(imgs[i], params_transform)
        table.insert(imgs_transf, new_img)
    end

    return {imgs_transf, label}
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

    -- concatenate data
    local imgs_tensor = torch.FloatTensor(batchSize, opt.seq_length,
                                          3, opt.inputRes, opt.inputRes):fill(0)

    -- images
    for i=1, batchSize do
        for j=1, opt.seq_length do
            imgs_tensor[i][j]:copy(sample[i][1][j])
        end
    end

    -- labels
    local labels_tensor = torch.IntTensor(batchSize):fill(0)
    for i=1, batchSize do
        labels_tensor[i] = sample[i][2]
    end

    collectgarbage()

    return imgs_tensor, labels_tensor
end
