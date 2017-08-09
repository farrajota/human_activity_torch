--[[
    List of available models to load.
]]

require 'nn'

if not nn.NoBackprop then paths.dofile('modules/NoBackprop.lua') end

local network_list = {}

network_list['vgg16-lstm'] = paths.dofile('vgg16_lstm.lua')
network_list['vgg16-lstm2'] = paths.dofile('vgg16_lstm2.lua')
network_list['kps-lstm'] = paths.dofile('body_joints_lstm.lua')
network_list['vgg16-kps-lstm'] = paths.dofile('vgg16_body_joints_lstm.lua')

network_list['vgg16-convnet'] = paths.dofile('vgg16_convnet.lua')
network_list['kps-convnet'] = paths.dofile('body_joints_convnet.lua')
network_list['vgg16-kps-convnet'] = paths.dofile('vgg16_body_joints_convnet.lua')

return network_list