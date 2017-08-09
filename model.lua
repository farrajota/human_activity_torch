--[[
    Load model network into memory.
]]


--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------

-- Continuing an experiment where it left off
local features_net, kps_net, classifier_net
opt.iniEpoch = 1
if opt.continue or opt.branch ~= 'none' then
    local prevModel
    if paths.filep(opt.save .. '/optim.t7') then
        prevModel = opt.save .. '/model.t7'
    else
        prevModel = opt.save .. '/model_' .. epoch .. '.t7'
    end

    print('==> Loading model from: ' .. prevModel)
    features_net, kps_net, classifier_net, opt.params = unpack(torch.load(prevModel))
    opt.iniEpoch = epoch

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    features_net, kps_net, classifier_net, opt.params = unpack(torch.load(opt.loadModel))

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    -- load models
    local models_list = paths.dofile('models/init.lua')
    assert(models_list[opt.netType], 'Undefined model architecture: ' .. opt.netType)
    features_net, kps_net, classifier_net, opt.params = models_list[opt.netType]()
end


--------------------------------------------------------------------------------
-- Define criterion
--------------------------------------------------------------------------------

local criterion = nn.CrossEntropyCriterion()


--------------------------------------------------------------------------------
-- Convert to GPU or CPU
--------------------------------------------------------------------------------

print('Running on GPU: [' .. opt.nGPU .. ']')
if features_net then features_net:cuda() end
if kps_net then kps_net:cuda() end
classifier_net:cuda()
criterion:cuda()
opt.dataType = 'torch.CudaTensor'


--------------------------------------------------------------------------------
-- Config network to use multiple GPUs
--------------------------------------------------------------------------------

-- Use multiple gpus
if opt.GPU >= 1 and opt.nGPU > 1 then
    if torch.type(model) == 'nn.DataParallelTable' then
        if features_net then features_net = utils.loadDataParallel(features_net, opt.nGPU) end
        if kps_net then kps_net = utils.loadDataParallel(kps_net, opt.nGPU) end
    else
        if features_net then features_net = utils.makeDataParallelTable(features_net, opt.nGPU) end
        if kps_net then kps_net = utils.makeDataParallelTable(kps_net, opt.nGPU) end
    end
end


--------------------------------------------------------------------------------
-- Output
--------------------------------------------------------------------------------

return features_net, kps_net, classifier_net, criterion