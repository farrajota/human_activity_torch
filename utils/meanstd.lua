--[[
    Compute the mean/std of a dataset
]]


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