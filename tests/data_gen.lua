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
paths.dofile('../sample_batch.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'ucf_sports'
opt.rotate = 15
opt.scale = 0.2
opt.rotRate = 0.5
opt.batchSize = 2
niters = 4
mode = 'train'
plot_results = true

local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]

for i=1, niters do
    print(('Iter %d/%d'):format(i, niters))
    if i==6 then
        a=1  -- stop debugger here
    end
    local input, label = getSampleBatch(loader, opt.batchSize, true)

    if plot_results then
        a = {}
        for ibatch=1, opt.batchSize do 
          disp.image(input[ibatch])
          print('label: ' .. label[ibatch])
        end
    end
end

print('Data fetching successfully finished.')