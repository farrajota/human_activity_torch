--[[
    Data sampling functions.
]]


-------------------------------------------------------------------------------
-- Data loading functions
-------------------------------------------------------------------------------

local function get_db_loader(name)
    local dbc = require 'dbcollection'

    local dbloader
    local str = string.lower(name)
    if str == 'ucfsports' then
        dbloader = dbc.load{name='ucfsports', task='recognition', data_dir=opt.data_dir}
    elseif str == 'ucf101' then
        dbloader = dbc.load{name='ucf101', task='recognition', data_dir=opt.data_dir}
    else
        error(('Invalid dataset name: %s. Available datasets: ucfsports | ucf101.'):format(name))
    end
    return dbloader
end

------------------------------------------------------------------------------------------------------------

local function loader_ucf_sports(set_name)
    local utils = require 'dbcollection.utils'
    local ascii2str = utils.string_ascii.convert_ascii_to_str
    local unpad_list = utils.pad.unpad_list

    local dbloader = get_db_loader('ucfsports')

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- number of categories
    local num_activities = dbloader:size(set_name, 'activities')[1]

    -- data loader function
    local data_loader = function(idx)

        -- select a random activity
        local iactivity = math.random(1, num_activities)

        -- select a random video from the selected activity
        local ivideo = dbloader:get(set_name, 'list_videos_per_activity', iactivity)

        -- get data from the selected video
        local data = dbloader:object(set_name, ivideo, true)[1]

        local label = dbloader.object(set_name, idx)[1][4]

        -- random start from the video sequence
        local idx_ini = math.random(1, data[5] - opt.seq_length)

        local imgs = {}
        for i=1, opt.seq_length do
            local filename = idx_ini --ascii2str(data[1])[1]
            local img_filename = paths.concat(dbloader.data_dir, filename)

            table.insert(imgs, image.load(img_filename, 3, 'float'))
        end

        return imgs, label
    end

    return {
        loader = data_loader,
        size = set_size,
        num_activities = num_activities
    }
end

------------------------------------------------------------------------------------------------------------

local function loader_ucf_101(set_name)
    local string_ascii = require 'dbcollection.utils.string_ascii'
    local ascii2str = string_ascii.convert_ascii_to_str

    local dbloader = get_db_loader('ucf101')

    -- number of samples per train/test sets
    local set_size = dbloader:size(set_name)[1]

    -- TODO

    return {
        loader = data_loader,
        size = set_size,
        num_keypoints = nJoints
    }
end

------------------------------------------------------------------------------------------------------------

local function fetch_loader_dataset(name, mode)
    local str = string.lower(name)
    if str == 'ucfsports' then
        return loader_ucf_sports(mode)
    elseif str == 'ucf101' then
        return loader_ucf_101(mode)
    else
        error(('Invalid dataset name: %s. Available datasets: ucfsports | ucf101.'):format(name))
    end
end
------------------------------------------------------------------------------------------------------------

function select_dataset_loader(name)
    assert(name)

    return {
        train = fetch_loader_dataset(name, 'train01'),
        test = fetch_loader_dataset(name, 'test01')
    }
end