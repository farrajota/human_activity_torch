--[[
    Load the VGG16 network from disk.
]]

local function load_network(conv_layers_only)
    local conv_layers_only = conv_layers_only or false

    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')

    local net = torch.load(paths.concat(filepath, 'model_vgg16.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_vgg16.t7'))

    if conv_layers_only then
        net:remove(net:size()) -- remove logsoftmax layer
        net:remove(net:size()) -- remove 3rd linear layer
        net:remove(net:size()) -- remove 2nd dropout layer
        net:remove(net:size()) -- remove 2nd last relu layer
        net:remove(net:size()) -- remove 2nd linear layer
        net:remove(net:size()) -- remove 1st dropout layer
        net:remove(net:size()) -- remove 1st relu layer
        net:remove(net:size()) -- remove 1st linear layer
        net:remove(net:size()) -- remove view layer

        params.feat_size = 512
    else
        net:remove(net:size()) -- remove logsoftmax layer
        net:remove(net:size()) -- remove 3rd linear layer

        params.feat_size = 4096
    end

    return net, params
end

return load_network