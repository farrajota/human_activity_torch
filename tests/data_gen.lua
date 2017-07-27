--[[
    Test the data generator
]]

require 'torch'
require 'paths'
disp = require 'display'

torch.manualSeed(4)
torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('../projectdir.lua')
utils = paths.dofile('../util/utils.lua')
paths.dofile('../data.lua')
paths.dofile('../transform.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'ucfsports'
opt.rotate = 15
opt.scale = 0.2
opt.rotRate = 0.5
niters = 1
mode = 'train'
plot_results = false

local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]

for i=1, niters do
    print(('Iter %d/%d'):format(i, niters))
    if i==6 then
        a=1  -- stop debugger here
    end
    local input, label = getSampleBatch_new(loader, opt.batchSize)

    if plot_results then
        a = {}
        disp.image(input)
        print('label: ' .. label)
    end
end

print('Data fetching successfully finished.')