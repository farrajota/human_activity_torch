--[[
    Script for testing a human activity estimator.

    Available/valid datasets: UCF Sports Action.
--]]


require 'paths'
require 'torch'
require 'string'

local tnt = require 'torchnet'


--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
--------------------------------------------------------------------------------

paths.dofile('configs.lua')

-- load model from disk
load_model('test')

-- set local vars
local lopt = opt

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end


--------------------------------------------------------------------------------
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
        ordered = true,
        init    = function(threadid)
                    require 'torch'
                    require 'torchnet'
                    opt = lopt
                    paths.dofile('data.lua')
                    paths.dofile('transform.lua')
                    torch.manualSeed(threadid+opt.manualSeed)
                  end,
        closure = function()

            -- setup data loader
            local data_loader = select_dataset_loader(opt.dataset, mode)
            local loader = data_loader[mode]

            -- setup dataset iterator
            return tnt.ListDataset{
                list = torch.range(1, nSamples):long(),
                load = function(idx)
                    local input, parts, center, scale, normalize = getSampleTest(loader, idx)
                    return {input, parts, center, scale, normalize}
                end
            }:batch(1, 'include-last')
        end,
    }
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local meters = {
    test_err = tnt.AverageValueMeter(),
    test_accu = tnt.AverageValueMeter(),
}

function meters:reset()
    self.test_err:reset()
    self.test_accu:reset()
end

local loggers = {
    test = Logger(paths.concat(opt.save,'test.log'), opt.continue)
}

loggers.test:setNames{'Test Loss', 'Test acc.'}

loggers.test.showPlot = false

-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
    print('\n*********************************************************')
    print(('Start testing the network on the %s dataset: '):format(opt.dataset))
    print('*********************************************************')
end


-- copy sample to GPU buffer:
local inputs = cast(torch.Tensor())
local targets, center, scale, normalize, t_matrix

engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();
    inputs:resize(state.sample[1]:size() ):copy(state.sample[1])
    parts   = state.sample[2][1]
    center  = state.sample[3][1]
    scale   = state.sample[4][1]
    normalize = state.sample[5][1]

    state.sample.input  = inputs
end


local predictions, distances = {}, {}
local coords = torch.FloatTensor(2, num_keypoints, nSamples):fill(0)

engine.hooks.onForward= function(state)
    xlua.progress(state.t, nSamples)

    -- compute the classification accuracy of the network
    --local acc = accuracy(state.network.output, state.sample.target)
    --
    --meters.test_err:add(state.criterion.output)
    --meters.test_accu:add(acc)

    collectgarbage()
end


engine.hooks.onEnd= function(state)
    -- Display accuracy
end


--------------------------------------------------------------------------------
-- Test the model
--------------------------------------------------------------------------------

engine:test{
    network  = model,
    iterator = getIterator('test')
}

print('\nTest script complete.')