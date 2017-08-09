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
opt.seq_length = 20
opt.GPU = 1
opt.nGPU = 1

local model_features, model_kps, model_classifier, criterion, params = paths.dofile('../model.lua')
opt.params = params

print('==> Features network:')
print(model_features)

print('==> kps network:')
print(model_kps)

print('==> Classifier network:')
print(model_classifier)

print('############################')
print('# process some input data')
print('############################')


local batch_feats_imgs = {}
for ibatch=1, opt.batchSize do
    local seq_feats = {}
    for i=1, opt.seq_length  do
        local img =  torch.Tensor(1,3,224,224):uniform():cuda()
        local features = model_features:forward(img)
        table.insert(seq_feats, features)
    end
    -- convert table into a single tensor
    table.insert(batch_feats_imgs, nn.Unsqueeze(1):cuda():forward(nn.JoinTable(1):cuda():forward(seq_feats)))
end
-- convert table into a single tensor
local inputs = nn.JoinTable(1):cuda():forward(batch_feats_imgs)

-- classify the sequence of images
local res = model_classifier:forward(inputs)

print(#res)