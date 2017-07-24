--[[
    Download VGG-16 and VGG-19 pretrained modeld on imagenet.

    source (vgg-16): https://gist.github.com/ksimonyan/211839e770f7b538e2d8
    source (vgg-19): https://gist.github.com/ksimonyan/3785162f95cd2d5fee77
]]


require 'paths'
require 'torch'
require 'loadcaffe'
paths.dofile('../projectdir.lua')


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Fast-RCNN download pretrained models.')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-save_path',  projectDir .. '/data', 'Experiment ID')
cmd:text()

local opt = cmd:parse(arg or {})
local savepath = paths.concat(opt.save_path, 'pretrained_models')

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end

print('==> Downloading VGG 16 and 19 models...')

local url1 = 'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel'
local url2 = 'https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/c3ba00e272d9f48594acef1f67e5fd12aff7a806/VGG_ILSVRC_16_layers_deploy.prototxt'
local url3 = 'http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_19_layers.caffemodel'
local url4 = 'https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/bb2b4fe0a9bb0669211cf3d0bc949dfdda173e9e/VGG_ILSVRC_19_layers_deploy.prototxt'

-- file names
local vgg_16_filename_model = paths.concat(savepath, 'VGG_ILSVRC_16_layers.caffemodel')
local vgg_16_filename_proto = paths.concat(savepath, 'VGG_ILSVRC_16_layers_deploy.prototxt')
local vgg_19_filename_model = paths.concat(savepath, 'VGG_ILSVRC_19_layers.caffemodel')
local vgg_19_filename_proto = paths.concat(savepath, 'VGG_ILSVRC_19_layers_deploy.prototxt')

-- download file
-- url1
if not paths.filep(vgg_16_filename_model) then
  local command = ('wget -O %s %s'):format(vgg_16_filename_model, url1)
  os.execute(command)
end
-- url2
if not paths.filep(vgg_16_filename_proto) then
  local command = ('wget -O %s %s'):format(vgg_16_filename_proto, url2)
  os.execute(command)
end
-- url3
if not paths.filep(vgg_19_filename_model) then
  local command = ('wget -O %s %s'):format(vgg_19_filename_model, url3)
  os.execute(command)
end
-- url4
if not paths.filep(vgg_19_filename_proto) then
  local command = ('wget -O %s %s'):format(vgg_19_filename_proto, url4)
  os.execute(command)
end

-- load network
local model_vgg16 = loadcaffe.load(vgg_16_filename_proto, vgg_16_filename_model, 'cudnn')
local model_vgg19 = loadcaffe.load(vgg_19_filename_proto, vgg_19_filename_model, 'cudnn')

-- model's parameters
local params = {}
params.mean = {103.939, 116.779, 123.68}
params.pixel_scale = 255.0
params.colourspace = 'bgr'
params.num_feats = 512
params.stride = 16 --pixels

-- save to memory
torch.save(paths.concat(savepath, 'model_vgg16.t7'), model_vgg16)
torch.save(paths.concat(savepath, 'model_vgg19.t7'), model_vgg19)
torch.save(paths.concat(savepath, 'parameters_vgg16.t7'), params)
torch.save(paths.concat(savepath, 'parameters_vgg19.t7'), params)

collectgarbage()

print('Done.')