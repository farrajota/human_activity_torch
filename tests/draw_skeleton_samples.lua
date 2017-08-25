--[[
    Test loading the hms (body joints predictor) networks (LSTM + ConvNet3D).
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

disp = require 'display'

paths.dofile('../projectdir.lua')
torch.manualSeed(14)
torch.setdefaulttensortype('torch.FloatTensor')

utils = paths.dofile('../util/utils.lua')
paths.dofile('../util/draw.lua')
paths.dofile('../util/heatmaps.lua')

paths.dofile('../data.lua')
paths.dofile('../sample_batch.lua')


local opts = paths.dofile('../options.lua')
opt = opts.parse(arg)
opt.netType = 'hms-convnet3d_8'
opt.dataset = 'ucf_sports'
opt.nFeats = 256
opt.num_activities = 10
opt.nLayers = 2
opt.batchSize = 6
opt.seq_length = 16
opt.GPU = 1
opt.nGPU = 1
opt.rotRate=1
opt.scale = 0
opt.db = 'lsp'
mode = 'train'

-- load model
local model_features, model_hms, model_classifier, criterion = paths.dofile('../model.lua')

-- fetch data
local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]


print('==> Features network:')
print(model_features)

print('==> hms network:')
print(model_hms)

print('==> Classifier network:')
print(model_classifier)

print('############################')
print('# Generate some input data')
print('############################')

local imgs_hms, imgs_feats, labels_tensor = getSampleBatch(loader, opt.batchSize, false)

local batch_coords = {}
local batch_heatmaps = {}
local batch_skeletons = {}
local batch_img = {}
for ibatch=1, opt.batchSize do
    local seq_heatmaps = {}
    local seq_coords = {}
    local seq_skeletons = {}
    local seq_img = {}
    for i=1, imgs_hms:size(2) do
        local img = imgs_hms[ibatch][i]
        local img_cuda = img:view(1, unpack(img:size():totable())):cuda()  -- extra dimension for cudnn batchnorm
        local heatmaps = model_hms:forward(img_cuda):float()
        heatmaps[heatmaps:lt(0)] = 0  -- clamp negatives values to 0
        local coord_hms = torch.mul(getPreds(heatmaps), 4)
        local img_skeleton = drawSkeleton(img, heatmaps[1], coord_hms[1], opt.db)
        table.insert(seq_heatmaps, heatmaps)
        table.insert(seq_coords, coord_hms)
        table.insert(seq_skeletons, img_skeleton)
        table.insert(seq_img, img)
    end
    -- convert table into a single tensor
    table.insert(batch_heatmaps, seq_heatmaps)
    table.insert(batch_coords, seq_coords)
    table.insert(batch_skeletons, seq_skeletons)
    table.insert(batch_img, seq_img)
end

c = drawImgHeatmapSingle(batch_img[1][16], batch_heatmaps[1][16]:squeeze())

for ibatch=1, opt.batchSize do
  disp.image(batch_skeletons[ibatch])
end


