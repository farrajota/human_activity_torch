--[[
    List of available models to load.
]]

require 'nn'

if not nn.NoBackprop then paths.dofile('modules/NoBackprop.lua') end

local network_list = {}
paths.dofile('test/init.lua')(model_list)

return network_list