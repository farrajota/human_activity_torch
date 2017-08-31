--[[
    Train and test all LSTM architectures (vgg16).
]]


paths.dofile('../projectdir.lua')
local logger_filename = paths.concat(projectDir,'exp', 'ucf_sports', 'test_lstm_architectures_vgg16.log')
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
        use_center_crop = 'false',

        -- train options
        optMethod = 'adam',
        LR = 1e-4,
        nThreads = 2,
        nEpochs = 10,
        trainIters = 300,
        testIters = 100,
        seq_length = 10,
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
    -- vgg16
    -- no center crop
    {expID = 'architectures-lstm-test1', netType = 'vgg16-hms-lstm-v1', use_center_crop = 'false'},
    {expID = 'architectures-lstm-test2', netType = 'vgg16-hms-lstm-v2', use_center_crop = 'false'},
    {expID = 'architectures-lstm-test3', netType = 'vgg16-hms-lstm-v3', use_center_crop = 'false'},

    -- center crop
    {expID = 'architectures-lstm-test7', netType = 'vgg16-hms-lstm-v1', use_center_crop = 'true'},
    {expID = 'architectures-lstm-test8', netType = 'vgg16-hms-lstm-v2', use_center_crop = 'true'},
    {expID = 'architectures-lstm-test9', netType = 'vgg16-hms-lstm-v3', use_center_crop = 'true'},
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

