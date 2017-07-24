--[[
    Data transform functions.
]]

require 'torch'
require 'image'


paths.dofile('util/img.lua')
paths.dofile('util/eval.lua')
paths.dofile('util/Logger.lua')
paths.dofile('util/store.lua')
paths.dofile('util/draw.lua')
paths.dofile('util/utils.lua')


--[[ Function for data augmentation, randomly samples on a normal distribution. ]]--
local function rnd(x)
    return math.max(-2*x, math.min(2*x, torch.randn(1)[1]*x))
end


-------------------------------------------------------------------------------
-- Transform data for one object sample
-------------------------------------------------------------------------------

function transform_data(img, keypoints, center, scale, nJoints)

    -- inits
    local rot = 0 -- set rotation to 0

    -- Do rotation + scaling
    if mode == 'train' then
        -- Scale and rotation augmentation
        scale = scale * (2 ^ rnd(opt.scale))
        rot = rnd(opt.rotate)
        if torch.uniform() <= opt.rotRate then
            rot = 0
        end
    end

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1, nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            local new_kp = transform(keypoints[i], center, scale, rot, opt.outputRes)
            drawGaussian(heatmap[i], new_kp, opt.hmGauss)
        end
    end

    -- Do image augmentation/normalization
    if mode == 'train' then
        -- Flipping
        if torch.uniform() < .5 then
            img_transf = flip(img_transf)
            heatmap = shuffleLR(flip(heatmap))
        end
        -- color augmentation
        if opt.colourjit then
            local opts_jit = {brightness = 0.4,
                              contrast = 0.4,
                              saturation = 0.4}
            img_transf = t.ColorJitter(opts_jit)(img_transf)
        else
            img_transf[1]:mul(torch.uniform(0.6, 1.4)):clamp(0,1)
            img_transf[2]:mul(torch.uniform(0.6, 1.4)):clamp(0,1)
            img_transf[3]:mul(torch.uniform(0.6, 1.4)):clamp(0,1)
        end
    end

    -- output
    return img_transf, heatmap
end

------------------------------------------------------------------------------------------------------------

--[[ Transform the data for accuracy evaluation ]]
function transform_data_test(img, keypoints, center, scale, nJoints)
    -- inits
    local rot = 0 -- set rotation to 0

    -- Crop image + craft heatmap
    local img_transf = crop2(img, center, scale, rot, opt.inputRes)
    local heatmap = torch.zeros(nJoints, opt.outputRes, opt.outputRes)
    for i = 1, nJoints do
        if keypoints[i][1] > 1 then -- Checks that there is a ground truth annotation
            --drawGaussian(heatmap[i], mytransform(torch.add(keypoints[i],1), c, s, r, opt.outputRes), 1)
            drawGaussian(heatmap[i], mytransform(keypoints[i], center, scale, rot, opt.outputRes), opt.hmGauss or 1)
        end
    end

    -- output: input, label, center, scale, normalize
    return img_transf, keypoints:narrow(2,1,2), center, scale, normalize
end


-------------------------------------------------------------------------------
-- Get a batch of data samples
-------------------------------------------------------------------------------

local function fetch_single_data(data_loader, idx)
    assert(data_loader)
    assert(idx)

    local img, keypoints, center, scale, nJoints = data_loader.loader(idx)

    if type(img) ~= 'table' then
        local imgs_t, heatmaps_t = transform_data(img, keypoints, center, scale, nJoints)
        return {imgs_t, heatmaps_t }
    else
        return {}
    end
end

------------------------------------------------------------------------------------------------------------

local function get_batch(data_loader, batchSize)
    assert(data_loader)
    assert(batchSize)

    local size = data_loader.size
    local max_attempts = 30
    local batchData, idxUsed = {}, {}

    for i=1, batchSize do
        local data = {}
        local attempts = 0
        while not next(data) do
            local idx = torch.random(1, size)
            if not idxUsed[idx] then
                data = fetch_single_data(data_loader, idx)
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


-------------------------------------------------------------------------------
-- Compute mean/std for the dataset
-------------------------------------------------------------------------------

function ComputeMeanStd(loader)
    assert(loader)

    print('Preparing meanstd cache...')

    local tnt = require 'torchnet'

    local nSamples = 1000
    local batchSize = 1

    print('Estimating the mean (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')

    -- setup dataset iterator
    local iter = tnt.ListDataset{  -- replace this by your own dataset
        list = torch.range(1, nSamples):long(),
        load = function(idx)
            local input, _ = getSampleBatch(loader, batchSize)
            return input[1]
        end
    }:iterator()

    local tm = torch.Timer()
    local meanEstimate = {0,0,0}
    local idx = 1
    xlua.progress(0, nSamples)
    for img in iter() do
        for j=1, 3 do
            meanEstimate[j] = meanEstimate[j] + img[j]:mean()
        end
        idx = idx + 1
        if idx%50==0 then xlua.progress(idx, nSamples) end
    end
    xlua.progress(nSamples, nSamples)
    for j=1, 3 do
        meanEstimate[j] = meanEstimate[j] / nSamples
    end
    local mean = meanEstimate


    print('Estimating the std (per-channel, shared for all pixels) over ' .. nSamples .. ' randomly sampled training images')
    local stdEstimate = {0,0,0}
    idx = 1
    xlua.progress(0, nSamples)
    for img in iter() do
        for j=1,3 do
            stdEstimate[j] = stdEstimate[j] + img[j]:std()
        end
        idx = idx + 1
        if idx%50==0 then xlua.progress(idx, nSamples) end
    end
    xlua.progress(nSamples, nSamples)
    for j=1,3 do
        stdEstimate[j] = stdEstimate[j] / nSamples
    end
    local std = stdEstimate

    local cache = {
        mean = mean,
        str = str
    }

    print('Time to estimate:', tm:time().real)
    return cache
end