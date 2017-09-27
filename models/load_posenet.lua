--[[
    Load body joint estimator network from disk.
]]


local function load_network()
    local filepath = paths.concat(projectDir, 'data', 'pretrained_models')
    local hg_net = torch.load(paths.concat(filepath, 'posenet.t7'))

    local nparts = 14

    local net = nn.Sequential()
    net:add(hg_net)
    net:add(nn.SelectTable(-1))
    local params = {
      pixel_scale = 1,
      dims = {nparts, 64, 64},
      feat_size = nparts
    }
    return net, params
end

return load_network