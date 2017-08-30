--[[
    Logging functions for testing.
]]

--[[
    Train and test all architectures in models/ dir.
]]


require 'optim'

if not projectDir then paths.dofile('../projectdir.lua') end

------------------------------------------------------------------------------------------------------------

local function exec_command(command)
    print('\n*****************************')
    print('*** Executing command:')
    print('*****************************\n')
    print(command)
    print('\n==============================')
    print('==============================\n')
    print('\n')
    os.execute(command)
end

------------------------------------------------------------------------------------------------------------

local function read_log_file(opt)
    local filename = paths.concat(projectDir, 'exp', opt.dataset, opt.expID, 'Evaluation_full.log')
    if paths.filep(filename) then
        local file = io.open(filename, 'r')
        local stats
        for line in file:lines() do
            stats = string.split(line, '\t')
        end
        file:close()
        return stats
    end
    return nil
end

------------------------------------------------------------------------------------------------------------

local function setup_logger(filename)
    assert(filename, 'Must input a file name+path.')

    local Logger = optim.Logger

    if not paths.dirp(paths.dirname(filename)) then
        print('Creating dir: ' .. paths.dirname(filename))
        os.execute('mkdir -p ' .. paths.dirname(filename))
    end
    logger = optim.Logger(filename, false)
    logger:setNames{'ExperimentID','Top-1 accuracy (%)','Top-5 accuracy (%)', 'Average Precision'}

    -- store test stats to a file
    function logger:log(opt)
        assert(opt)
        local stats = read_log_file(opt)
        self:add{opt.expID, tonumber(stats[1]), tonumber(stats[2]), tonumber(stats[3])}
    end

    -- display options to screen
    logger.exec_command = exec_command

    return logger
end

------------------------------------------------------------------------------------------------------------

return setup_logger
