--[[
    Train and test a Resnet18 + LSTM network to classify a sequence of images.
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
    expID = 'resnet18-lstm-test',
    dataset = 'ucf_sports',

    -- model
    netType = 'resnet18-lstm',
    nFeats = 256,
    nLayers = 2,

    -- data
    inputRes = 256,
    scale = .25,
    rotate = 30,
    rotRate = .5,

    -- train options
    optMethod = 'adam',
    nThreads = 4,
    nEpochs = 10,
    trainIters = 300,
    testIters = 100,
    seq_length = 30,
    batchSize = 4,
    grad_clip = 10,
    snapshot = 5,
    nGPU = 1,
    continue = 'false',
    saveBest = 'true',
    clear_buffers = 'true',

    -- test options
    test_progressbar = 'false',
    test_load_best = 'true'
}

-- concatenate options fields to a string
local str_args = ''
for k, v in pairs(opts) do
    str_args = str_args .. ('-%s %s '):format(k, v)
end

-- train network
--exec_command(('th train.lua %s'):format(str_args))

-- test network
exec_command(('th test.lua %s'):format(str_args))
