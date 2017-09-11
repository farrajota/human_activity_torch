--[[
    Load the ResNet network from disk.
]]


local filepath = paths.concat(projectDir, 'data', 'pretrained_models')

------------------------------------------------------------------------------------------------------------

local function load_resnet18()
    local net = torch.load(paths.concat(filepath, 'model_resnet-18.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_resnet-18.t7'))

    net:remove(net:size()) -- remove linear layer
    --net:remove(net:size()) -- remove view

    params.feat_size = 512

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_resnet32()
    local net = torch.load(paths.concat(filepath, 'model_resnet-32.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_resnet-32.t7'))

    net:remove(net:size()) -- remove linear layer
    --net:remove(net:size()) -- remove view

    params.feat_size = 512

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_resnet50()
    local net = torch.load(paths.concat(filepath, 'model_resnet-50.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_resnet-50.t7'))
    params.num_feats = 2048

    net:remove(net:size()) -- remove linear layer
    --net:remove(net:size()) -- remove view

    params.feat_size = 2048

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_resnet101()
    local net = torch.load(paths.concat(filepath, 'model_resnet-101.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_resnet-101.t7'))

    net:remove(net:size()) -- remove linear layer
    --net:remove(net:size()) -- remove view

    params.feat_size = 2048

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_resnet152()
    local net = torch.load(paths.concat(filepath, 'model_resnet-152.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_resnet-152.t7'))

    net:remove(net:size()) -- remove linear layer
    --net:remove(net:size()) -- remove view

    params.feat_size = 2048

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_resnet200()
    local net = torch.load(paths.concat(filepath, 'model_resnet-200.t7'))
    local params = torch.load(paths.concat(filepath, 'parameters_resnet-200.t7'))

    net:remove(net:size()) -- remove linear layer
    --net:remove(net:size()) -- remove view

    params.feat_size = 2048

    return net, params
end

------------------------------------------------------------------------------------------------------------

local function load_network(netType)
    local netType = netType or 'resnet50'
    if netType == 'resnet18' then
        return load_resnet18()
    elseif netType == 'resnet32' then
        return load_resnet32()
    elseif netType == 'resnet50' then
        return load_resnet50()
    elseif netType == 'resnet101' then
        return load_resnet101()
    elseif netType == 'resnet152' then
        return load_resnet152()
    elseif netType == 'resnet200' then
        return load_resnet200()
    else
        error('Invalid residual network: ' .. netType)
    end
end

------------------------------------------------------------------------------------------------------------

return load_network