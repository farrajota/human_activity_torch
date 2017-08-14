--[[
    List of available models to load.
]]

require 'nn'

if not nn.NoBackprop then paths.dofile('modules/NoBackprop.lua') end

local network_list = {}

network_list['vgg16-lstm'] = paths.dofile('vgg16_lstm.lua')
network_list['vgg16-lstm2'] = paths.dofile('vgg16_lstm2.lua')
network_list['kps-lstm'] = paths.dofile('kps_lstm.lua')
network_list['vgg16-kps-lstm'] = paths.dofile('vgg16_kps_lstm.lua')

network_list['vgg16-convnet3d'] = paths.dofile('vgg16_convnet3d.lua')
network_list['vgg16-convnet3d_2'] = paths.dofile('vgg16_convnet3d_2.lua')
network_list['kps-convnet3d'] = paths.dofile('kps_convnet3d.lua')
network_list['vgg16-kps-convnet3d'] = paths.dofile('vgg16_kps_convnet3d.lua')

return network_list