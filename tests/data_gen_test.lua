--[[
    Test the data generator
]]

require 'torch'
require 'paths'
require 'xlua'
disp = require 'display'

torch.manualSeed(4)
torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('../projectdir.lua')
utils = paths.dofile('../util/utils.lua')
paths.dofile('../data.lua')
paths.dofile('../sample_batch.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'ucf_sports'
opt.rotate = 15
opt.scale = 0.2
opt.rotRate = 0.5
opt.batchSize = 2
opt.seq_length = 30
opt.same_transform = true
opt.process_input_heatmap = false
opt.process_input_feats = true
opt.use_center_crop = false
niters = 1000
mode = 'test'
plot_results = false
opt.params = torch.load('./data/pretrained_models/parameters_vgg16.t7')

local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]

print('==> Test data samples')
for i=1, niters do
    --print(('Iter %d/%d'):format(i, niters))
    if i==183 then
        a=1  -- stop debugger here
    end
    local idx = torch.random(1, loader.num_videos)
    local input_hms, input_feats, label = getSampleTest(loader, idx)
    xlua.progress(i, niters)
end

print('Data fetching successfully finished.')