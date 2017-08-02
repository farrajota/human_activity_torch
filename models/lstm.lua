--[[
    LSTM network to classify a sequence of images.
]]


require 'nn'
require 'cudnn'

------------------------------------------------------------------------------------------------------------

local function load_network(input_size, num_feats, num_activities, num_layers, batchSize, seq_length)
    local view = nn.View(batchSize * seq_length, -1)
    local lstm = nn.Sequential()
    lstm:add(nn.Contiguous())
    lstm:add(cudnn.LSTM(input_size, num_feats, num_layers, true))
    lstm:add(nn.Contiguous())
    lstm:add(view)
    lstm:add(nn.Linear(num_feats, num_activities))
    return lstm, view
end

------------------------------------------------------------------------------------------------------------

return load_network