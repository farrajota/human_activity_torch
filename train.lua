--[[
    Script for training a human activty estimator network.
--]]

require 'paths'
require 'torch'
require 'string'
require 'optim'

local tnt = require 'torchnet'
local Logger = optim.Logger


--------------------------------------------------------------------------------
-- Load configs (data, model, criterion, optimState)
--------------------------------------------------------------------------------

print('==> (1/3) Load configurations: ')
paths.dofile('configs.lua')

-- load model + criterion
print('==> (2/3) Load/create network: ')
load_model('train')

utils.print_model_to_txt(paths.concat(opt.save, 'architecture.txt'),
                         {
                             {'==> Features network:', model_features},
                             {'==> Body Joint hms network:', model_hms},
                             {'==> Classifier network:', model_classifier}
                         })


-- set local vars
local lopt = opt
local nBatchesTrain = opt.trainIters
local nBatchesTest = opt.testIters

-- convert modules to a specified tensor type
local function cast(x) return x:type(opt.dataType) end

print('==> (3/3) Train the network on the dataset: ' .. opt.dataset)

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
                    paths.dofile('sample_batch.lua')
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
                    local input_hms, input_feats, label = getSampleBatch(loader, opt.batchSize, mode=='train')
                    return {
                        input_hms = input_hms,
                        input_feats = input_feats,
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

local timers = {
   batchTimer = torch.Timer(),
   dataTimer = torch.Timer(),
   epochTimer = torch.Timer(),
}

local meters = {
    train_conf = tnt.ConfusionMeter{k = opt.num_activities},
    test_conf = tnt.ConfusionMeter{k = opt.num_activities},
    test = tnt.AverageValueMeter(),
    train = tnt.AverageValueMeter(),
    train_clerr = tnt.ClassErrorMeter{topk = {1,5},accuracy=true},
    clerr = tnt.ClassErrorMeter{topk = {1,5},accuracy=true},
    ap = tnt.APMeter(),
}

function meters:reset()
    self.train_conf:reset()
    self.test_conf:reset()
    self.test:reset()
    self.train:reset()
    self.train_clerr:reset()
    self.clerr:reset()
    self.ap:reset()
end

local loggers = {
    test = Logger(paths.concat(opt.save,'test.log'), opt.continue),
    train = Logger(paths.concat(opt.save,'train.log'), opt.continue),
    full_train = Logger(paths.concat(opt.save,'full_train.log'), opt.continue),
    
    train_conf = Logger(paths.concat(opt.save, 'train_confusion.log'), opt.continue),
    test_conf = Logger(paths.concat(opt.save, 'test_confusion.log'), opt.continue),
}

loggers.test:setNames{'Test Loss', 'Test acc.', 'Test mAP'}
loggers.train:setNames{'Train Loss', 'Train acc.'}
loggers.full_train:setNames{'Train Loss', 'Train accuracy'}
loggers.train_conf:setNames{'Train confusion matrix'}
loggers.test_conf:setNames{'Test confusion matrix'}

loggers.test.showPlot = false
loggers.train.showPlot = false
loggers.full_train.showPlot = false
loggers.train_conf.showPlot = false
loggers.test_conf.showPlot = false


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
    state.network:training()  -- ensure the model is set to training mode
    timers.epochTimer:reset()
end


-- copy sample to GPU buffer:
local inputs, targets = cast(torch.Tensor()), cast(torch.Tensor())
local input_features
engine.hooks.onSample = function(state)
    cutorch.synchronize(); collectgarbage();

    --------
    local function process_inputs(model, input)
        local features = {}
        if model then
            local batch_feats_imgs = {}
            for ibatch=1, opt.batchSize do
                local seq_feats = {}
                for i=1, opt.seq_length do
                    local img = input[ibatch][i]
                    local img_cuda = img:view(1, unpack(img:size():totable())):cuda()  -- extra dimension for cudnn batchnorm
                    local features = model:forward(img_cuda)
                    table.insert(seq_feats, features)
                end
                -- convert table into a single tensor
                table.insert(batch_feats_imgs, nn.Unsqueeze(1):cuda():forward(nn.JoinTable(1):cuda():forward(seq_feats)))
            end
            -- convert table into a single tensor
            features = nn.JoinTable(1):cuda():forward(batch_feats_imgs)
        end
        collectgarbage()
        collectgarbage()
        return features
    end
    --------


    local inputs_features, inputs_hms = {}, {}
    if model_hms then
        if model_features then inputs_features = process_inputs(model_features, state.sample.input_feats[1]) end
        if model_hms then
            inputs_hms = process_inputs(model_hms, state.sample.input_hms[1])
            inputs_hms[inputs_hms:lt(0)]=0
        end
    else
        local batch_features = {}
        for ibatch=1, opt.batchSize do
            inputs:resize(state.sample.input_feats[1][ibatch]:size() ):copy(state.sample.input_feats[1][ibatch])
            local features = model_features:forward(inputs)
            if not input_features then
                input_features = cast(torch.Tensor(opt.batchSize, unpack(features:size():totable())))
            end
            input_features[ibatch]:copy(features)
        end
        inputs_features = input_features
        collectgarbage()
        collectgarbage()
    end

    if opt.flatten then
        if model_hms then
            inputs_hms = inputs_hms:view(opt.batchSize, opt.seq_length, -1)
        end
    end

    if model_features and model_hms then
        state.sample.input = {inputs_features, inputs_hms}
    elseif model_features then
        state.sample.input = inputs_features
    elseif model_hms then
        state.sample.input = inputs_hms
    else
        error('Invalid network type: ' .. opt.netType)
    end

    -- copy data to targets
    targets:resize(state.sample.target[1]:size() ):copy(state.sample.target[1])
    if string.find(opt.netType, 'lstm') then
        state.sample.target = targets:view(-1)
    elseif string.find(opt.netType, 'convnet') then
        state.sample.target = targets[{{},{1}}]:squeeze(2):contiguous()
    else
        error('Invalid network type: ' .. opt.netType)
    end

    timers.dataTimer:stop()
    timers.batchTimer:reset()
end


engine.hooks.onForward = function(state)
   if not state.training then
      xlua.progress(state.t, nBatchesTest)
      if state.t == 94 then
        aqui=1
        end
   end
end


engine.hooks.onUpdate = function(state)
    timers.dataTimer:reset()
    timers.dataTimer:resume()
end


engine.hooks.onForwardCriterion = function(state)
    if state.training then
        meters.train_conf:add(state.network.output,state.sample.target)
        meters.train:add(state.criterion.output)
        meters.train_clerr:add(state.network.output,state.sample.target)
        if opt.verbose then
            print(string.format('epoch[%d/%d][%d/%d][batch=%d][seq=%d] - loss: %2.4f; top-1 err: ' ..
                                '%2.2f; top-5 err: %2.2f; lr = %2.2e;  DataLoadingTime: %0.5f; ' ..
                                'forward-backward time: %0.5f', state.epoch+1, state.maxepoch,
                                state.t+1, nBatchesTrain, opt.batchSize, opt.seq_length, meters.train:value(),
                                100-meters.train_clerr:value{k = 1}, 100-meters.train_clerr:value{k = 5},
                                state.config.learningRate, timers.dataTimer:time().real,
                                timers.batchTimer:time().real))
        else
            xlua.progress(state.t+1, nBatchesTrain)
        end

        loggers.full_train:add{state.criterion.output}
    else
        meters.test_conf:add(state.network.output,state.sample.target)
        meters.clerr:add(state.network.output,state.sample.target)
        meters.test:add(state.criterion.output)
        local tar = torch.ByteTensor(#state.network.output):fill(0)
        for k=1,state.sample.target:size(1) do
            local id = state.sample.target[k]
            tar[k][id]=1
        end
        meters.ap:add(state.network.output,tar)
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

    print("Epoch Train Loss:" ,meters.train:value(),"Total Epoch time: ",timers.epochTimer:time().real)
    print("Accuracy: Top 1%", meters.train_clerr:value{k = 1} .. '%')
    print("Accuracy: Top 5%", meters.train_clerr:value{k = 5} .. '%')
    -- measure test loss and error:
    loggers.train:add{meters.train:value(),meters.train_clerr:value()[1]}
    local tr = optim.ConfusionMatrix(opt.activities)
    tr.mat = meters.train_conf:value()
    loggers.train_conf:add{tr:__tostring__()} -- output the confusion matrix as a string
    if opt.printConfusion then
        print(tr)
    else
        tr:updateValids();
        print('+ average row correct: ' .. (tr.averageValid*100) .. '%')
        print('+ average rowUcol correct (VOC measure): ' .. (tr.averageUnionValid*100) .. '%')
        print('+ global correct: ' .. (tr.totalValid*100) .. '%')
    end
    meters:reset()
    state.t = 0


    ---------------------
    -- test the network
    ---------------------

    local accuracy_top1
    if nBatchesTest > 0 then
        print('\n**********************************************')
        print(('Test network (epoch = %d/%d)'):format(state.epoch, state.maxepoch))
        print('**********************************************')
        engine:test{
            network   = model_classifier,
            iterator  = getIterator('test'),
            criterion = criterion,
        }

        loggers.test:add{meters.test:value(),meters.clerr:value()[1],meters.ap:value():mean()}
        print("Test Loss" , meters.test:value())
        print("Accuracy: Top 1%", meters.clerr:value{k = 1} .. '%')
        print("Accuracy: Top 5%", meters.clerr:value{k = 5} .. '%')
        print("mean AP:",meters.ap:value():mean())
        accuracy_top1 = meters.clerr:value{k = 1}

        local ts = optim.ConfusionMatrix(opt.activities)
        ts.mat = meters.test_conf:value()
        loggers.test_conf:add{ts:__tostring__()} -- output the confusion matrix as a string

        if opt.printConfusion then
            print(ts)
        else
            ts:updateValids();
            print('+ average row correct: ' .. (ts.averageValid*100) .. '%')
            print('+ average rowUcol correct (VOC measure): ' .. (ts.averageUnionValid*100) .. '%')
            print('+ global correct: ' .. (ts.totalValid*100) .. '%')
        end
    end

    --------------------------------
    -- save model snapshots to disk
    --------------------------------

    storeModel(model_features, model_hms, state.network, state.config, state.epoch, opt)

    ------------------------------------
    -- save best accuracy model to disk
    ------------------------------------

    if accuracy_top1 > test_best_accu and opt.saveBest then
        test_best_accu = accuracy_top1
        storeModelBest(model_features, model_hms, state.network, opt)
    end

    timers.epochTimer:reset()
    state.t = 0
end


--------------------------------------------------------------------------------
-- Train the model
--------------------------------------------------------------------------------

print('==> Train network model')
engine:train{
    network   = model_classifier,
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