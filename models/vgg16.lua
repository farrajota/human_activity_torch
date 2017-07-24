--[[
    Load VGG16 network.
]]


require 'nn'

------------------------------------------------------------------------------------------------------------

local function SelectFeatsDisableBackprop(net)
    local features = net
    features:remove(features:size()) -- remove logsoftmax layer
    features:remove(features:size()) -- remove 3rd linear layer
    return nn.NoBackprop(features)
end

------------------------------------------------------------------------------------------------------------

local function load_network()
    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')

    local net = torch.load(paths.concat(filepath, 'model_vgg16.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_vgg16.t7'))
    params.feat_size = 4096

    local features = SelectFeatsDisableBackprop(net), params
end

------------------------------------------------------------------------------------------------------------

return load_network