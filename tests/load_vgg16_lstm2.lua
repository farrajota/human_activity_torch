--[[
    Test loading the vgg16 + lstm model
]]

require 'torch'
require 'paths'
require 'string'
require 'nn'
require 'nngraph'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'

torch.manualSeed(4)
torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('../projectdir.lua')
utils = paths.dofile('../util/utils.lua')

local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.dataset = 'ucf_sports'
opt.netType = 'vgg16-lstm2'
opt.nFeats = 256
opt.num_activities = 10
opt.nLayers = 2
opt.batchSize = 2
opt.seq_length = 10
opt.GPU = 1
opt.nGPU = 1

local model, criterion = paths.dofile('../model.lua')

print(model)

print('############################')
print('# process some input data')
print('############################')

local input = torch.Tensor(opt.batchSize, opt.seq_length,3,224,224):uniform():cuda()

local res = model:forward(input)

print(#res)