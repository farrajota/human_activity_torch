--[[
    Train and test a LSTM network to classify a sequence of images.
]]

local function exec_command(command)
    print('\n')
    print('Executing command: ' .. command)
    print('\n')
    os.execute(command)
end

------------------------------------------------------------------------------------------------------------

local opts = {
    -- experiment id
    expID = 'vgg16-lstm-test',
    dataset = 'ucf_sports',

    -- model
    netType = 'vgg16-lstm',
    nFeats = 256,
    nLayers = 2,

    -- data
    inputRes = 256,
    scale = .25,
    rotate = 30,
    rotRate = .5,

    -- train options
    optMethod = 'adam',
    nThreads = 2,
    nEpochs = 10,
    trainIters = 300,
    testIters = 100,
    seq_length = 25,
    batchSize = 4,
    grad_clip = 10,
    snapshot = 5,
    nGPU = 1,
    continue = 'false',
    saveBest = 'true',
    clear_buffers = 'true',

    -- test options
    test_progressbar = 'false',
    test_load_best = 'false'
}

-- concatenate options fields to a string
local str_args = ''
for k, v in pairs(opts) do
    str_args = str_args .. ('-%s %s '):format(k, v)
end

local str_cuda
if opts.nGPU <= 1 then
    str_cuda = 'CUDA_VISIBLE_DEVICES=0'
else
    str_cuda = 'CUDA_VISIBLE_DEVICES=0,1'
end

-- train network
exec_command(('%s th train.lua %s'):format(str_cuda, str_args))

-- test network
exec_command(('%s th test.lua %s'):format(str_cuda, str_args))
