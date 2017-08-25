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

nparts = 14
opt.params = {
  pixel_scale = 1,
  dims = {nparts, 64, 64},
  feat_size = nparts
}

-- fetch data
local data_loader = select_dataset_loader(opt.dataset, mode)
local loader = data_loader[mode]


local imgs_hms, imgs_feats, labels_tensor = getSampleBatch(loader, opt.batchSize, false)



c = drawImgHeatmapSingle(batch_img[1][16], batch_heatmaps[1][16]:squeeze())

for ibatch=1, opt.batchSize do
  disp.image(batch_skeletons[ibatch])
end


