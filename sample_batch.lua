--[[
    Data transform functions.
]]

require 'torch'
require 'image'

paths.dofile('utils/img.lua')


------------------------------------------------------------------------------------------------------------

--[[ Function for data augmentation, randomly samples on a normal distribution. ]]--
local function rnd(x)
    return math.max(-2*x, math.min(2*x, torch.randn(1)[1]*x))
end


-------------------------------------------------------------------------------
-- Transform data for one object sample
-------------------------------------------------------------------------------

function transform_data(img, params)
    assert(imgs)
    assert(params)

    -- Crop image
    local img_transf = crop2(img.img, img.center, img.scale, params.rot, opt.inputRes)

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

    return imgs_resized, label
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

function getSampleBatch_new(data_loader, batchSize, is_train)
    assert(data_loader)
    assert(batchSize)
    assert(is_train)

    -- get batch data
    local sample = get_batch(data_loader, batchSize, is_train)

    -- concatenate data
    local imgs_tensor
    local imgs_tensor = nn.JoinTable(1):forward(sample[1])

    return imgs_tensor, label
end

------
function getSampleBatch(data_loader, batchSize)
    assert(data_loader)

    local batchSize = batchSize or opt.batchSize or 1

    -- get batch data
    local sample = get_batch(data_loader, batchSize)

    -- concatenate data
    local imgs_tensor = torch.FloatTensor(batchSize,
                                          sample[1][1]:size(1),
                                          sample[1][1]:size(2),
                                          sample[1][1]:size(3)):fill(0)
    local heatmaps_tensor = torch.FloatTensor(batchSize,
                                              sample[1][2]:size(1),
                                              sample[1][2]:size(2),
                                              sample[1][2]:size(3)):fill(0)

    for i=1, batchSize do
        imgs_tensor[i]:copy(sample[i][1])
        heatmaps_tensor[i]:copy(sample[i][2])
    end

    collectgarbage()

    return imgs_tensor, heatmaps_tensor
end

------------------------------------------------------------------------------------------------------------

function getSampleTest(data_loader, idx)

    -- set rotation to 0
    local rot = 0

    -- Load image + keypoints + other data
    local img, keypoints, center, scale, nJoints, normalize = data_loader.loader(idx)

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1,nJoints do
        -- Checks that there is a ground truth annotation
        if keypoints[i][1] > 1 then
            drawGaussian(heatmap[i], mytransform(keypoints[i], center, scale, rot, opt.outputRes), opt.hmGauss or 1)
        end
    end

    -- output: input, label, center, scale, normalize
    return img_transf, keypoints:narrow(2,1,2), center, scale, normalize
end

------------------------------------------------------------------------------------------------------------

function getSampleBenchmark(data_loader, idx)

    -- set rotation to 0
    local rot = 0

    -- Load image + keypoints + other data
    local img, keypoints, center, scale, nJoints, normalize = data_loader.loader(idx)

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)

    -- output: input, label, center, scale, normalize
    return img_transf, center, scale, normalize
end
