if not projectDir then
    paths.dofile('../projectdir.lua')
end

------------------------------------------------------------------------------------------------------------

local function load_logger(logger_filename)
    return paths.dofile('../util/logger.lua')(logger_filename)
end

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

return load_logger, get_configs