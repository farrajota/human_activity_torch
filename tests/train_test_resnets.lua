--[[
    Train and test all architectures in models/ dir.
]]


paths.dofile('../projectdir.lua')
local logger_filename = paths.concat(projectDir,'exp', 'ucf_sports', 'test_resnets.log')
local logger = paths.dofile('../util/logger.lua')(logger_filename)

------------------------------------------------------------------------------------------------------------

local function get_configs()
    return {
        -- experiment id
        dataset = 'ucf_sports',

        -- model
        nFeats = 512,
        nLayers = 2,

        -- data
        inputRes = 256,
        scale = .25,
        rotate = 30,
        rotRate = .5,
        colorjit = .3,
        same_transform = 'true',
        use_center_crop = 'true',

        -- train options
        optMethod = 'adam',
        LR = 1e-4,
        nThreads = 6,
        nEpochs = 10,
        trainIters = 300,
        testIters = 100,
        seq_length = 32,
        batchSize = 4,
        grad_clip = 10,
        snapshot = 0,
        nGPU = 1,
        continue = 'false',
        saveBest = 'true',
        clear_buffers = 'true',

        -- test options
        test_progressbar = 'false',
        test_load_best = 'true',
    }
end

------------------------------------------------------------------------------------------------------------

local test_opts = {
    {expID = 'resnet-lstm-test1', netType = 'resnet18-lstm'},
    {expID = 'resnet-lstm-test2', netType = 'resnet32-lstm'},
    {expID = 'resnet-lstm-test3', netType = 'resnet50-lstm'},
    {expID = 'resnet-lstm-test4', netType = 'resnet101-lstm'},
    {expID = 'resnet-lstm-test5', netType = 'resnet152-lstm'},
    {expID = 'resnet-lstm-test6', netType = 'resnet200-lstm'},

    {expID = 'resnet-lstm-test7', netType = 'resnet18-lstm', same_transform= 'false', use_center_crop = 'true'},
    {expID = 'resnet-lstm-test8', netType = 'resnet18-lstm', same_transform= 'false', use_center_crop = 'false'},
    {expID = 'resnet-lstm-test9', netType = 'resnet18-lstm', same_transform= 'true', use_center_crop = 'false'},
}

for i, test_opt in ipairs(test_opts) do
    print('\n=======================================================')
    print(('Starting train+test configuration: %d/%d'):format(i, #test_opts))
    print('=======================================================\n')

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
    logger.exec_command(('th train.lua %s'):format(str_args))

    -- test network
    logger.exec_command(('th test.lua %s'):format(str_args))

    -- log best accuracy
    logger:log(opts)
end

