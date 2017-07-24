--[[
    List of available models to load.
]]


local model_list = {}

model_list['rnn']     = paths.dofile('rnn.lua')  -- SML net
model_list['lstm']    = paths.dofile('lstm.lua')  -- SML net
model_list['convnet'] = paths.dofile('convnet.lua')  -- SML net

return model_list