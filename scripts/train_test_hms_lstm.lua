--[[
    Train and test the Pose + LSTM network.
]]


local load_logger, get_configs = paths.dofile('opts.lua')

local logger_filename = paths.concat(projectDir,'exp', 'ucf_sports', 'hms_lstm.log')
local logger = load_logger(logger_filename)

local opts = {
    {expID = 'hms-lstm', netType = 'hms-lstm'},
}

for i, test_opt in ipairs(opts) do
    print('\n=======================================================')
    print(('Starting train+test configuration: %d/%d'):format(i, #opts))
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

