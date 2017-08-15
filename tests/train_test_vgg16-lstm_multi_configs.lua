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

local function get_configs()
    return {
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
        LR = 2.5e-4,
        nThreads = 2,
        nEpochs = 10,
        trainIters = 300,
        testIters = 100,
        seq_length = 25,
        batchSize = 4,
        grad_clip = 0,
        snapshot = 0,
        nGPU = 1,
        continue = 'false',
        saveBest = 'true',
        clear_buffers = 'true',

        -- test options
        test_progressbar = 'false',
        test_load_best = 'true'
    }
end

------------------------------------------------------------------------------------------------------------

local test_opts = {
    -- try different sequence lengths and batch sizes
    {expID = 'vgg16-lstm-test1', seq_length = 10, batchSize = 2},
    {expID = 'vgg16-lstm-test2', seq_length = 20, batchSize = 2},
    {expID = 'vgg16-lstm-test3', seq_length = 30, batchSize = 2},
    {expID = 'vgg16-lstm-test4', seq_length = 10, batchSize = 4},
    {expID = 'vgg16-lstm-test5', seq_length = 20, batchSize = 4},
    {expID = 'vgg16-lstm-test6', seq_length = 30, batchSize = 4},
    {expID = 'vgg16-lstm-test7', seq_length = 10, batchSize = 16},
    {expID = 'vgg16-lstm-test8', seq_length = 20, batchSize = 16},
    {expID = 'vgg16-lstm-test9', seq_length = 30, batchSize = 16},
    {expID = 'vgg16-lstm-test10', seq_length = 10, batchSize = 32},
    {expID = 'vgg16-lstm-test11', seq_length = 20, batchSize = 32},
    {expID = 'vgg16-lstm-test12', seq_length = 30, batchSize = 32},

    -- try different nFeats and nLayers
    {expID = 'vgg16-lstm-test13', seq_length = 10, batchSize = 4, nFeats = 256, nLayers = 1,},
    {expID = 'vgg16-lstm-test14', seq_length = 10, batchSize = 4, nFeats = 512, nLayers = 1,},
    {expID = 'vgg16-lstm-test15', seq_length = 10, batchSize = 4, nFeats = 256, nLayers = 2,},
    {expID = 'vgg16-lstm-test16', seq_length = 10, batchSize = 4, nFeats = 512, nLayers = 2,},
    {expID = 'vgg16-lstm-test17', seq_length = 10, batchSize = 4, nFeats = 256, nLayers = 3,},
    {expID = 'vgg16-lstm-test18', seq_length = 10, batchSize = 4, nFeats = 512, nLayers = 3,},

    -- try different learning rates and grad clipping
    {expID = 'vgg16-lstm-test19', seq_length = 10, batchSize = 4, LR = 1e-3,   grad_clip = 0,},
    {expID = 'vgg16-lstm-test20', seq_length = 10, batchSize = 4, LR = 2.5e-4, grad_clip = 0,},
    {expID = 'vgg16-lstm-test21', seq_length = 10, batchSize = 4, LR = 1e-4,   grad_clip = 0,},
    {expID = 'vgg16-lstm-test22', seq_length = 10, batchSize = 4, LR = 1e-3,   grad_clip = 10,},
    {expID = 'vgg16-lstm-test23', seq_length = 10, batchSize = 4, LR = 2.5e-4, grad_clip = 10,},
    {expID = 'vgg16-lstm-test24', seq_length = 10, batchSize = 4, LR = 1e-4,   grad_clip = 10,},

    -- try different data augmentation
    {expID = 'vgg16-lstm-test25', seq_length = 10, batchSize = 4, rotRate = .5, scale = .25, rotate = 30,},
    {expID = 'vgg16-lstm-test26', seq_length = 10, batchSize = 4, rotRate = .5, scale = .25, rotate = 15,},
    {expID = 'vgg16-lstm-test27', seq_length = 10, batchSize = 4, rotRate = .5, scale = .25, rotate = 0,},
    {expID = 'vgg16-lstm-test28', seq_length = 10, batchSize = 4, rotRate = .5, scale = .15, rotate = 30,},
    {expID = 'vgg16-lstm-test29', seq_length = 10, batchSize = 4, rotRate = .5, scale = 0,   rotate = 30,},
    {expID = 'vgg16-lstm-test30', seq_length = 10, batchSize = 4, rotRate = 0,  scale = 0,   rotate = 0,},
}


for i, test_opt in ipairs(test_opts) do
    print(('Starting train+test configuration: %d/%d'):format(i, #test_opts))

    -- set train + test options
    local opts = get_configs()
    for k, v in pairs(test_opt) do
        opts[k] = v
    end

    -- concatenate options fields to a string
    local str_args = ''
    for k, v in pairs(opts) do
        str_args = str_args .. ('-%s %s '):format(k, v)
    end

    -- train network
    exec_command(('th train.lua %s'):format(str_args))

    -- test network
    exec_command(('th test.lua %s'):format(str_args))
end

