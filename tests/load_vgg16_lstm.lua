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
opt.netType = 'vgg16-lstm'
opt.nFeats = 256
opt.num_activities = 10
opt.nLayers = 2
opt.batchSize = 2
opt.seq_length = 20
opt.GPU = 1
opt.nGPU = 1

local model_features, model_kps, model_classifier, criterion, opt.params = paths.dofile('../model.lua')

print('==> Features network:')
print(model_features)

print('==> Features network:')
print(model_kps)

print('==> Classifier network:')
print(model_classifier)

print('############################')
print('# process some input data')
print('############################')


local inputs = {}
for ibatch=1, opt.batchSize do
    local batch_feats = {}
    for i=1, opt.seq_length do
        local img = torch.Tensor(1,3,224,224):uniform():cuda()
        local features = model_features:forward(img)
        table.insert(batch_feats, features)
    end
    table.insert(inputs, nn.JoinTable(1):cuda():forward(batch_feats))
end
inputs = nn.JoinTable(1):cuda():forward(inputs)

-- classify the sequence of images
local res = model_classifier:forward(inputs)

print(#res)