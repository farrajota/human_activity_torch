--[[
    Download ResNet pretrained models on imagenet.

    source: https://github.com/facebook/fb.resnet.torch/tree/master/pretrained
]]


require 'paths'
require 'torch'
paths.dofile('../projectdir.lua')


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Torch-7 Fast-RCNN download pretrained models.')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-save_dir',   projectDir .. '/data', 'Experiment ID')
cmd:text()

local opt = cmd:parse(arg or {})
local savepath = paths.concat(opt.save_path, 'pretrained_models')

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end

print('==> Downloading resnet-18, resnet-32, resnet-50, resnet-101, resnet-152, resnet-200 models...')

local url1 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-18.t7'
local url2 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-34.t7'
local url3 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-50.t7'
local url4 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-101.t7'
local url5 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-152.t7'
local url6 = 'https://d2j0dndfm35trm.cloudfront.net/resnet-200.t7'

-- file names
local filename_model18 = paths.concat(savepath, 'model_resnet-18.t7')
local filename_model32 = paths.concat(savepath, 'model_resnet-32.t7')
local filename_model50 = paths.concat(savepath, 'model_resnet-50.t7')
local filename_model101 = paths.concat(savepath, 'model_resnet-101.t7')
local filename_model152 = paths.concat(savepath, 'model_resnet-152.t7')
local filename_model200 = paths.concat(savepath, 'model_resnet-200.t7')

-- download file
-- url1
if not paths.filep(filename_model18) then
  local command = ('wget -O %s %s'):format(filename_model18, url1)
  os.execute(command)
end
-- url2
if not paths.filep(filename_model32) then
  local command = ('wget -O %s %s'):format(filename_model32, url2)
  os.execute(command)
end
-- url3
if not paths.filep(filename_model50) then
  local command = ('wget -O %s %s'):format(filename_model50, url3)
  os.execute(command)
end
-- url4
if not paths.filep(filename_model101) then
  local command = ('wget -O %s %s'):format(filename_model101, url4)
  os.execute(command)
end
-- url5
if not paths.filep(filename_model152) then
  local command = ('wget -O %s %s'):format(filename_model152, url5)
  os.execute(command)
end
-- url6
if not paths.filep(filename_model200) then
  local command = ('wget -O %s %s'):format(filename_model200, url6)
  os.execute(command)
end

-- model's parameters
local params = {}
params.mean = {0.485, 0.456, 0.406}
params.std = {0.229, 0.224, 0.225}
params.pixel_scale = 1.0
params.colourspace = 'rgb'
params.num_feats = 512
params.stride = 32 --pixels

-- save to memory
torch.save(paths.concat(savepath, 'parameters_resnet-18.t7'), params)
torch.save(paths.concat(savepath, 'parameters_resnet-32.t7'), params)

params.num_feats = 2048
torch.save(paths.concat(savepath, 'parameters_resnet-50.t7'), params)
torch.save(paths.concat(savepath, 'parameters_resnet-101.t7'), params)
torch.save(paths.concat(savepath, 'parameters_resnet-152.t7'), params)
torch.save(paths.concat(savepath, 'parameters_resnet-200.t7'), params)

collectgarbage()

print('Done.')