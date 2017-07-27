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
    local num_activities = dbloader:size(set_name, 'activities')

    -- data loader function
    local data_loader = function(idx)

        -- select a random activity
        local iactivity = math.random(1, num_activities)

        -- select a random video from the selected activity
        local video_ids = unpad_list(dbloader:get(set_name, 'list_videos_per_activity', iactivity))
        local video = videos[math.random(video_ids[1], video_ids[#video_ids])] + 1  -- set to 1-index

        -- fetch all object ids belonging to the video
        local obj_ids = unpad_list(dbloader:get(set_name, 'list_object_ids_per_video', video) + 1)  -- set to 1-index

        -- get data from the selected video
        local data = dbloader:object(set_name, obj_ids, true)

        local label = iactivity

        -- random start from the video sequence
        local idx_ini = math.random(1, #obj_ids - opt.seq_length)

        local imgs, bboxes = {}, {}
        for i=1, opt.seq_length do
            local filename = ascii2str(data[i][1])
            local img_filename = paths.concat(dbloader.data_dir, filename)

            table.insert(imgs, image.load(img_filename, 3, 'float'))
            table.insert(bboxes, data[i][2])
        end

        return imgs, bboxes, label
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