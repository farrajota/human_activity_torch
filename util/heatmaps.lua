--[[
    Heatmap related functions.
]]


------------------------------------------------------------------------------------------------------------

local function fitParabola(x1,x2,x3,y1,y2,y3)
  local x1_sqr = x1*x1
  local x2_sqr = x2*x2
  local x3_sqr = x3*x3

  local div = (x1_sqr-x1*(x2+x3)+x2*x3)*(x2-x3)
  local a = (x1*(y2-y3)-x2*(y1-y3)+x3*(y1-y2))/div
  local b = (x1_sqr*(y2-y3)-x2_sqr*(y1-y3)+x3_sqr*(y1-y2))/div

  return b/(2*a)
end

------------------------------------------------------------------------------------------------------------

local function fitParabolaAll(hms, coords)
    local preds = coords:clone():fill(0)
    local nparts = hms:size(2)
    local h,w = hms:size(3),hms:size(4)
    for i=1, hms:size(1) do
        for j=1, nparts do
            local x = {coords[i][j][1]-1, coords[i][j][1], coords[i][j][1]+1}
            local y = {coords[i][j][2]-1, coords[i][j][2], coords[i][j][2]+1}

            if x[1]>=1 and x[3]<=w and y[2]>=1 and y[2]<=h then
                preds[i][j][1] = fitParabola(x[1],x[2],x[3],hms[i][j][y[2]][x[1]],hms[i][j][y[2]][x[2]],hms[i][j][y[2]][x[3]])
            else
                preds[i][j][1]=x[2] -- skip parabola fitting for this coordinate
            end

            if y[1]>=1 and y[3]<=h and x[2]>=1 and x[2]<=w then
                preds[i][j][2] = fitParabola(y[1],y[2],y[3],hms[i][j][y[1]][x[2]],hms[i][j][y[2]][x[2]],hms[i][j][y[3]][x[2]])
            else
                preds[i][j][2]=y[2] -- skip parabola fitting for this coordinate
            end
        end
    end
    return preds
end

------------------------------------------------------------------------------------------------------------

function getPreds(hm)
    assert(hm:dim() == 4, 'Input must be 4-D tensor')
    --local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    --local preds = torch.repeatTensor(idx, 1, 1, 2):float()
    --preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hm:size(4) + 1 end)
    --preds[{{}, {}, 2}]:add(-1):div(hm:size(3)):floor():add(1)

    local max, idx = torch.max(hm:view(hm:size(1), hm:size(2), hm:size(3) * hm:size(4)), 3)
    local coords_peak = torch.repeatTensor(idx, 1, 1, 2):float()
    coords_peak[{{}, {}, 1}]:apply(function(x) if x%hm:size(4)==0 then return hm:size(4) else return x%hm:size(4) end end)
    coords_peak[{{}, {}, 2}]:div(hm:size(3)):ceil()
    local preds = fitParabolaAll(hm, coords_peak)

    return preds
end

------------------------------------------------------------------------------------------------------------

function getPredsFull(hms, center, scale)
    assert(hms)
    assert(center)
    assert(scale)

    if hms:dim() == 3 then hms = hms:view(1, hms:size(1), hms:size(2), hms:size(3)) end

    local preds, preds_tf
    if opt.subpixel_precision then
        -- Get locations of maximum activations (using sub-pixel precision)
        local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
        local coords_peak = torch.repeatTensor(idx, 1, 1, 2):float()
        coords_peak[{{}, {}, 1}]:apply(function(x) return x % hms:size(4) end)
        coords_peak[{{}, {}, 2}]:div(hms:size(3)):ceil()
        preds = fitParabolaAll(hms, coords_peak)

        -- Get transformed coordinates
        preds_tf = torch.zeros(preds:size())
        for i = 1,hms:size(1) do        -- Number of samples
            for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
                preds_tf[i][j] = transform(preds[i][j],center,scale,0,hms:size(3),true)
            end
        end
    else
        -- Get locations of maximum activations (OLD CODE)
        local max, idx = torch.max(hms:view(hms:size(1), hms:size(2), hms:size(3) * hms:size(4)), 3)
        preds = torch.repeatTensor(idx, 1, 1, 2):float()
        preds[{{}, {}, 1}]:apply(function(x) return (x - 1) % hms:size(4) + 1 end)
        preds[{{}, {}, 2}]:add(-1):div(hms:size(3)):floor():add(1)
        local predMask = max:gt(0):repeatTensor(1, 1, 2):float()
        preds:add(-.5):cmul(predMask):add(1)

        -- Get transformed coordinates
        preds_tf = torch.zeros(preds:size())
        for i = 1,hms:size(1) do        -- Number of samples
            for j = 1,hms:size(2) do    -- Number of output heatmaps for one sample
                preds_tf[i][j] = transformBenchmark(preds[i][j],center,scale,0,hms:size(3),true)
            end
        end
    end

    return preds, preds_tf
end
