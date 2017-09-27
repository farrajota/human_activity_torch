--[[
    Train and test all LSTM architectures (vgg16).
]]


paths.dofile('../projectdir.lua')
local logger_filename = paths.concat(projectDir,'exp', 'ucf_sports', 'test_lstm_architectures_final.log')
local logger = paths.dofile('../util/logger.lua')(logger_filename)

------------------------------------------------------------------------------------------------------------

local function get_configs()
    return {
        -- experiment id
        dataset = 'ucf_sports',

        -- model
        nFeats = 512,
        nLayers = 2,
        convert_cudnn = 'true',

        -- data
        inputRes = 256,
        scale = .10,
        rotate = 30,
        rotRate = .5,
        colorjit = .3,
        heatmap_jit = 6,
        dropout=.5,
        same_transform_heatmaps = 'true',
        same_transform_features = 'false',
        use_center_crop = 'true',

        -- train options
        optMethod = 'adam',
        LR = 1e-4,
        nThreads = 4,
        nEpochs = 30,
        trainIters = 300,
        testIters = 100,
        seq_length = 30,
        step=2,
        batchSize = 4,
        grad_clip = 10,
        snapshot = 0,
        nGPU = 1,
        continue = 'false',
        saveBest = 'true',
        clear_buffers = 'true',
        printConfusion = 'true',

        -- test options
        test_progressbar = 'false',
        test_load_best = 'true',
        test_seq_length = 90,
        test_step = 1,
        test_printConfusion = 'true',
    }
end

------------------------------------------------------------------------------------------------------------

local test_opts = {
    {expID = 'final-resnet50-lstm', netType = 'resnet50-lstm'},
    {expID = 'final-hms-lstm', netType = 'hms-lstm'},
    {expID = 'final-resnet50-hms-lstm', netType = 'resnet50-hms-lstm-v2'},
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

