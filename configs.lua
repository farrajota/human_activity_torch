--[[
    Loads necessary libraries and files.
]]


require 'paths'
require 'torch'
require 'xlua'
require 'optim'
require 'nn'
require 'nngraph'
require 'string'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('projectdir.lua') -- Project directory
paths.dofile('data.lua')
paths.dofile('models/modules/NoBackprop.lua')
paths.dofile('util/store.lua')
paths.dofile('util/meanstd.lua')
utils = paths.dofile('util/utils.lua')


-----------------------------------------------------------
-- Load options
-----------------------------------------------------------

if not opt then
    local opts = paths.dofile('options.lua')
    opt = opts.parse(arg)

    print('Saving everything to: ' .. opt.save)
    os.execute('mkdir -p ' .. opt.save)

    if opt.GPU >= 1 then
        cutorch.setDevice(opt.GPU)
    end

    if opt.branch ~= 'none' or opt.continue then
        -- Continuing training from a prior experiment
        -- Figure out which new options have been set

        if paths.filep(opt.save .. '/options.t7') then
            local setOpts = {}
            for i = 1,#arg do
                if arg[i]:sub(1,1) == '-' then table.insert(setOpts,arg[i]:sub(2,-1)) end
            end
            -- Where to load the previous options/model from
            if opt.branch ~= 'none' then
                opt.load = opt.expDir .. '/' .. opt.branch
            else
                opt.load = opt.expDir .. '/' .. opt.expID
            end

            -- Keep previous options, except those that were manually set
            local opt_ = opt
            opt = torch.load(opt_.load .. '/options.t7')
            opt.save = opt_.save
            opt.load = opt_.load
            opt.continue = opt_.continue
            for i = 1,#setOpts do
                opt[setOpts[i]] = opt_[setOpts[i]]
            end

            -- determine highest epoc and load corresponding model
            local last_epoch = torch.load(opt.load .. '/last_epoch.t7')
            epoch = last_epoch

        else
            epoch = 1
        end
    else
        epoch = 1
    end

    opt.epochNumber = epoch
    nEpochs = opt.nEpochs

    -- Training hyperparameters
    if not optimState then
        if type(opt.schedule)=='table' then
            local schedule = {}
            local schedule_id = 0
            for i=1, #opt.schedule do
                table.insert(schedule, {schedule_id+1, schedule_id+opt.schedule[i][1],
                    opt.schedule[i][2],
                    opt.schedule[i][3]})
                schedule_id = schedule_id+opt.schedule[i][1]
            end
            optimStateFn = function(epoch)
                for k, v in pairs(schedule) do
                    if v[1] <= epoch and v[2] >= epoch then
                        return {
                            learningRate = v[3],
                            learningRateDecay = opt.LRdecay,
                            momentum = opt.momentum,
                            dampening = 0.0,
                            weightDecay = v[4],
                            beta1 = opt.beta1,  -- adam
                            beta2 = opt.beta2,  -- adam
                            alpha = opt.alpha,  -- rmsprop
                            epsilon = opt.epsilon,  -- adam/rmsprop
                            end_schedule = (v[2]==epoch and 1) or 0
                        }
                    end
                end
                return optimState
            end
            -- determine the maximum number of epochs
            for k, v in pairs(schedule) do
                nEpochs = math.min(v[2])
            end
        else
            optimStateFn = function(epoch)
                return {
                    learningRate = opt.LR,
                    learningRateDecay = opt.LRdecay,
                    momentum = opt.momentum,
                    dampening = 0.0,
                    weightDecay = opt.weightDecay,
                    beta1 = opt.beta1,  -- adam
                    beta2 = opt.beta2,  -- adam
                    alpha = opt.alpha,  -- rmsprop
                    epsilon = opt.epsilon,  -- adam/rmsprop
                }
            end
        end
    end

    -- Random number seed
    if opt.manualSeed ~= -1 then
        torch.manualSeed(opt.manualSeed)
    else
        torch.seed()
    end

    -- Save options to experiment directory
    torch.save(opt.save .. '/options.t7', opt)
end


--------------------------------------------------------------------------------
-- Number of activities
--------------------------------------------------------------------------------

-- setup data loader
local data_loader = select_dataset_loader(opt.dataset)
local loader = data_loader['test']
opt.test_num_videos = loader.num_videos
loader = data_loader['train']
opt.num_activities = loader.num_activities
opt.num_videos = loader.num_videos


-----------------------------------------------------------
-- Functions
-----------------------------------------------------------

function load_model(mode)
    local str = string.lower(mode)
    if str == 'train' then
        -- Load model
        model_features, model_kps, model_classifier, criterion = paths.dofile('model.lua')

        if model_features then model_features:evaluate() end
        if model_kps then model_kps:evaluate() end
        model_classifier:training()
    elseif str == 'test' then
        -- load model
        print('Loading models from file: ' .. opt.load)
        model_features, model_kps, model_classifier, opt.params = unpack(torch.load(opt.load))

        if opt.GPU >= 1 then
            opt.dataType = 'torch.CudaTensor'  -- Use GPU
            if model_features then model_features:cuda() end
            if model_kps then model_kps:cuda() end
            model_classifier:cuda()

            -- convert to cuda
            if pcall(require, 'cudnn') then
                print('Converting model features+classifier backend to cudnn...')
                if model_features then cudnn.convert(model_features, cudnn):cuda() end
                if model_kps then cudnn.convert(model_kps, cudnn):cuda() end
                cudnn.convert(model_classifier, cudnn):cuda()
                print('Done.')
            end

            if model_features then
                if torch.type(model_features) == 'nn.DataParallelTable' then
                    model_features = utils.loadDataParallel(model_features, 1)
                end  -- load model into 1 GPU
            end
            if model_kps then
                if torch.type(model_features) == 'nn.DataParallelTable' then
                    model_kps = utils.loadDataParallel(model_kps, 1)
                end  -- load model into 1 GPU
            end
        else
            error('Undefined behaviour for non-GPU/cuda models.')
            --opt.dataType = 'torch.FloatTensor' -- Use CPU
            --model_classifier:float()
        end

        if model_features then model_features:evaluate() end
        if model_kps then model_kps:evaluate() end
        model_classifier:evaluate()
    else
        error(('Invalid mode: %s. mode must be either \'train\' or \'test\''):format(mode))
    end
end
