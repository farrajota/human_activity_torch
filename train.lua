--[[
    Script for training a human activty estimator network.
--]]

require 'paths'
require 'torch'
require 'string'

local tnt = require 'torchnet'


--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
--------------------------------------------------------------------------------

print('==> (1/4) Load configurations: ')
paths.dofile('configs.lua')

-- load model + criterion
print('==> (2/4) Load/create network: ')
load_model('train')

-- Compute the statistics of the images
print('==> (3/4) Computing dataset statistics (mean/std): ')
process_mean_std()

-- set local vars
local lopt = opt
--local dataset = select_dataset_loader(opt.dataset, 'train')
local nBatchesTrain = opt.trainIters
local nBatchesTest = opt.testIters

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end

print('==> (4/4) Train the network on the dataset: ' .. opt.dataset)

print('\n**********************')
print('Optimizer: ' .. opt.optMethod)
print('**********************\n')


--------------------------------------------------------------------------------
-- Setup data generator
--------------------------------------------------------------------------------

local function getIterator(mode)
    return tnt.ParallelDatasetIterator{
        nthread = opt.nThreads,
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

            -- number of iterations
            local nIters = (mode=='train' and opt.trainIters) or opt.testIters

            -- setup dataset iterator
            return tnt.ListDataset{
                list = torch.range(1, nIters):long(),
                load = function(idx)
                    local input, label = getSampleBatch(loader, opt.batchSize)
                    return {
                        input = input,
                        target = label
                    }
                end
            }:batch(1, 'include-last')
        end,
    }
end


--------------------------------------------------------------------------------
-- Setup torchnet engine/meters/loggers
--------------------------------------------------------------------------------

local meters = {
    train_err = tnt.AverageValueMeter(),
    train_accu = tnt.AverageValueMeter(),
    test_err = tnt.AverageValueMeter(),
    test_accu = tnt.AverageValueMeter(),
}

function meters:reset()
    self.train_err:reset()
    self.train_accu:reset()
    self.test_err:reset()
    self.test_accu:reset()
end

local loggers = {
    test = Logger(paths.concat(opt.save,'test.log'), opt.continue),
    train = Logger(paths.concat(opt.save,'train.log'), opt.continue),
    full_train = Logger(paths.concat(opt.save,'full_train.log'), opt.continue),
}

loggers.test:setNames{'Test Loss', 'Test acc.'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss', 'Train accuracy'}

loggers.test.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false


-- set up training engine:
local engine = tnt.OptimEngine()

engine.hooks.onStart = function(state)
    if state.training then
        state.config = optimStateFn(state.epoch+1)
        if opt.epochNumber>1 then
            state.epoch = math.max(opt.epochNumber, state.epoch)
        end
    end
end


engine.hooks.onStartEpoch = function(state)
    print('\n**********************************************')
    print(('Starting Train epoch %d/%d  %s'):format(state.epoch+1, state.maxepoch,  opt.save))
    print('**********************************************')
    state.config = optimStateFn(state.epoch+1)
    state.network:training() -- ensure the model is set to training mode
end


-- copy sample to GPU buffer:
local inputs, targets = cast(torch.Tensor()), cast(torch.Tensor())
engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();
    inputs:resize(state.sample.input[1]:size() ):copy(state.sample.input[1])
    targets:resize(state.sample.target[1]:size() ):copy(state.sample.target[1])

    state.sample.input  = inputs
    state.sample.target = utils.ReplicateTensor2Table(targets, opt.nOutputs)
end


engine.hooks.onForwardCriterion = function(state)
    if state.training then
        xlua.progress((state.t+1), nBatchesTrain)

        -- compute the PCK accuracy of the networks (last) output heatmap with the ground-truth heatmap
        local acc = accuracy(state.network.output, state.sample.target)

        meters.train_err:add(state.criterion.output)
        meters.train_accu:add(acc)
        loggers.full_train:add{state.criterion.output, acc}
    else
        xlua.progress(state.t, nBatchesTest)

        -- compute the PCK accuracy of the networks (last) output heatmap with the ground-truth heatmap
        local acc = accuracy(state.network.output, state.sample.target)

        meters.test_err:add(state.criterion.output)
        meters.test_accu:add(acc)
    end
end


--[[ Gradient clipping to try to prevent the gradient from exploding. ]]--
-- ref: https://github.com/facebookresearch/torch-rnnlib/blob/master/examples/word-language-model/word_lm.lua#L216-L233
local function clipGradients(grads, norm)
    local totalnorm = grads:norm()
    if totalnorm > norm then
        local coeff = norm / math.max(totalnorm, 1e-6)
        grads:mul(coeff)
    end
end
engine.hooks.onBackward = function(state)
    if opt.grad_clip > 0 then
        clipGradients(state.gradParams, opt.grad_clip)
    end
end


local test_best_accu = 0
engine.hooks.onEndEpoch = function(state)
    ---------------------------------
    -- measure test loss and error:
    ---------------------------------

    print(('Train Loss: %0.5f; Acc: %0.5f'):format(meters.train_err:value(),  meters.train_accu:value()))
    local tr_loss = meters.train_err:value()
    local tr_accuracy = meters.train_accu:value()
    loggers.train:add{tr_loss, tr_accuracy}
    meters:reset()
    state.t = 0


    ---------------------
    -- test the network
    ---------------------

    if nBatchesTest > 0 then
        print('\n**********************************************')
        print(('Test network (epoch = %d/%d)'):format(state.epoch, state.maxepoch))
        print('**********************************************')
        engine:test{
            network   = model,
            iterator  = getIterator('test'),
            criterion = criterion,
        }
        local ts_loss = meters.test_err:value()
        local ts_accuracy = meters.test_accu:value()
        loggers.test:add{ts_loss, ts_accuracy}
        print(('Test Loss: %0.5f; Acc: %0.5f'):format(meters.test_err:value(),  meters.test_accu:value()))

        --[[ Save model with the best accuracy ]]--
        if ts_accuracy > test_best_accu and opt.saveBest then
            storeModelBest(state.network.modules[1], opt)
            test_best_accu = ts_accuracy
        end
    end

    -----------------------------
    -- save model snapshots to disk
    -----------------------------

    storeModel(state.network.modules[1], state.config, state.epoch, opt)

    state.t = 0
end


--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------

print('==> Train network model')
engine:train{
    network   = model,
    iterator  = getIterator('train'),
    criterion = criterion,
    optimMethod = optim[opt.optMethod],
    config = optimStateFn(1),
    maxepoch = nEpochs
}


--------------------------------------------------------------------------------
-- Plot log graphs
--------------------------------------------------------------------------------

loggers.test:style{'+-', '+-'}; loggers.test:plot()
loggers.train:style{'+-', '+-'}; loggers.train:plot()
loggers.full_train:style{'-', '-'}; loggers.full_train:plot()

print('==> Script complete.')