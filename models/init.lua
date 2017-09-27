--[[
    List of available models to load.
]]

require 'nn'

if not nn.NoBackprop then paths.dofile('modules/NoBackprop.lua') end

local network_list = {}
paths.dofile('test/init.lua')(network_list)

network_list['resnet50-lstm'] = paths.dofile('test/resnet50_lstm.lua')
network_list['hms-lstm'] = paths.dofile('test/hms_lstm.lua')
network_list['resnet50-hms-lstm'] = paths.dofile('test/resnet50_hms_lstm_v2.lua')

return network_list