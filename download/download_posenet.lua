--[[
    Download the Human Pose Estimation network pretrained on the LSP+MPII datasets.
]]


require 'paths'
require 'torch'
paths.dofile('../projectdir.lua')


local cmd = torch.CmdLine()
cmd:text()
cmd:text('Download Pose pretrained model.')
cmd:text()
cmd:text(' ---------- General options ------------------------------------')
cmd:text()
cmd:option('-save_dir',   projectDir .. '/data', 'Experiment ID')
cmd:text()

local opt = cmd:parse(arg or {})
local savepath = paths.concat(opt.save_dir, 'pretrained_models')

-- create directory if needed
if not paths.dirp(savepath) then
    print('creating directory: ' .. savepath)
    os.execute('mkdir -p ' .. savepath)
end

print('==> Downloading PoseNet model...')

local url = 'https://www.dropbox.com/s/wlqiravxj5dh41d/posenet.t7?dl=0'

-- file names
local filename_model = paths.concat(savepath, 'posenet.t7')

-- download file
if not paths.filep(filename_model) then
  local command = ('wget -O %s %s'):format(filename_model, url)
  os.execute(command)
end

print('Done.')