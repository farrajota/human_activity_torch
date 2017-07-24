--[[
    Load model network into memory.
]]


--------------------------------------------------------------------------------
-- Load model
--------------------------------------------------------------------------------

-- Continuing an experiment where it left off
local model
opt.iniEpoch = 1
if opt.continue or opt.branch ~= 'none' then
    local prevModel
    if paths.filep(opt.save .. '/optim.t7') then
        prevModel = opt.save .. '/model.t7'
    else
        prevModel = opt.save .. '/model_' .. epoch .. '.t7'
    end

    print('==> Loading model from: ' .. prevModel)
    model = torch.load(prevModel)
    opt.iniEpoch = epoch

-- Or a path to previously trained model is provided
elseif opt.loadModel ~= 'none' then
    assert(paths.filep(opt.loadModel), 'File not found: ' .. opt.loadModel)
    print('==> Loading model from: ' .. opt.loadModel)
    model = torch.load(opt.loadModel)

-- Or we're starting fresh
else
    print('==> Creating model from file: models/' .. opt.netType .. '.lua')
    -- load models
    local models_list = paths.dofile('models/init.lua')
    assert(models_list[opt.netType], 'Undefined model architecture: ' .. opt.netType)
    model = models_list[opt.netType]()
end


--------------------------------------------------------------------------------
-- Define criterion
--------------------------------------------------------------------------------

local criterion = nn.CrossEntropyCriterion()


--------------------------------------------------------------------------------
-- Convert to GPU or CPU
--------------------------------------------------------------------------------

if opt.GPU >= 1 then
    print('Running on GPU: [' .. opt.nGPU .. ']')
    require 'cutorch'
    require 'cunn'
    model:cuda()
    criterion:cuda()

   -- require cudnn if available
    if pcall(require, 'cudnn') then
        cudnn.convert(model, cudnn):cuda()
        cudnn.benchmark = true
        if opt.cudnn_deterministic then
            model:apply(function(m) if m.setMode then m:setMode(1,1,1) end end)
        end
        print('Network has', #model:findModules'cudnn.SpatialConvolution', 'cudnn convolutions')
    end
    opt.dataType = 'torch.CudaTensor'
else
    print('Running on CPU')
    model:float()
    criterion:float()
    opt.dataType = 'torch.FloatTensor'
end


--------------------------------------------------------------------------------
-- Config network to use multiple GPUs
--------------------------------------------------------------------------------

local modelOut = nn.Sequential()

-- Use multiple gpus
if opt.GPU >= 1 and opt.nGPU > 1 then
    if torch.type(model) == 'nn.DataParallelTable' then
        modelOut:add(utils.loadDataParallel(model, opt.nGPU))
    else
        modelOut:add(utils.makeDataParallelTable(model, opt.nGPU))
    end
else
    modelOut:add(model)
end

local function cast(x) return x:type(opt.data_type) end

cast(modelOut)


--------------------------------------------------------------------------------
-- Output
--------------------------------------------------------------------------------

return modelOut, criterion