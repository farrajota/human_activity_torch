--[[
    Demo the activity estimation network on a dataset.
--]]


require 'torch'
require 'image'
disp = require 'display'


print('==> (1/3) Load configurations: ')
paths.dofile('configs.lua')

-- load model from disk
print('==> (2/3) Load network from disk: ')
load_model('test')

-- select random video
local mode = 'test'
local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]
local activities = loader.activities

local video_ids = {}
if next(opt.demo_video_ids) then
    video_ids = opt.demo_video_ids
else
    video_ids = torch.Tensor(opt.demo_nvideos):random(1, loader.num_videos):totable()
end
print(('==> (3/3) Select %d videos from %s'):format(#video_ids, opt.dataset)


for ivideo, video_idx in ipairs(video_ids) do
    print((' > Processing Video %d/%d: '):format(ivideo, #video_ids))

    print('   - Fetch video image sequence...')
    local input_hms, input_feats, label = getSampleTest(loader, idx)
    local num_imgs_seq = input_hms:size(2)

    -- process images features
    local inputs_features = {}
    if model_features then
        print('   - Process image features...')
        for i=1, num_imgs_seq do
            local img = input_feats[1][i]
            local img_cuda = img:view(1, unpack(img:size():totable())):cuda()  -- extra dimension for cudnn batchnorm
            local features = model_features:forward(img_cuda)
            table.insert(inputs_features, features)
        end
        -- convert table into a single tensor
        inputs_features = nn.JoinTable(1):cuda():forward(inputs_features)
        inputs_features = inputs_features:view(1, num_imgs_seq, -1)
    end

    -- process images body joints
    local inputs_hms = {}
    if model_hms then
        print('   - Process image body joints...')
        for i=1, num_imgs_seq do
            local img =  input_hms[1][i]
            local img_cuda = img:view(1, unpack(img:size():totable())):cuda()  -- extra dimension for cudnn batchnorm
            local hms = model_hms:forward(img_cuda)
            table.insert(inputs_hms, hms)
        end
        -- convert table into a single tensor
        inputs_hms = nn.JoinTable(1):cuda():forward(inputs_hms)
        inputs_hms = nn.Unsqueeze(1):cuda():forward(inputs_hms)
    end

    local input
    if model_features and model_hms then
        input = {inputs_features, inputs_hms}
    elseif model_features then
        input = inputs_features
    elseif model_hms then
        input = inputs_hms
    else
        error('Invalid network type: ' .. opt.netType)
    end

    --------------------------------------------------------------------------------
    -- Classify sequence of images
    --------------------------------------------------------------------------------

    print('   - Classify sequence...')
    local res = model_classifier:forward(input)
    local result = nn.Softmax():forward(res:float())


    --------------------------------------------------------------------------------
    -- Display video sequence (in browser)
    --------------------------------------------------------------------------------

    print('   - Plot results to browser')
    local top_n = 5  -- top5
    local state_win
    for i=1, input_hms:size(2) do
        local img = input_hms[1][i]

        -- fetch top-5 classification results
        local sorted_results, sorted_idx = torch.sort(result, true)

        -- add text to image
        for j=1, top_n do
            local act_res, act_idx = sorted_results[j], sorted_idx[j]
            local activity_name = activities[act_idx]
            img = image.drawText(img,  -- image
                                 ("%s: %0.5f"):format(activity_name, act_res),  -- text
                                 3+(j-1)*5, 3,  -- (x,y)
                                 {color = {0, 0, 0}, size = 3})
        end

        -- display image
        state_win = disp.image(img, {win=state_win,
                                     title=('Video %d, frame %d (%s)'):format(video_idx, i, activities[label])})
        sys.sleep(0.1)
    end
end
