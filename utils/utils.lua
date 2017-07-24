--[[
    Usefull utility functions for managing networks.
]]


local ffi=require 'ffi'

------------------------------------------------------------------------------------------------------------

local function MSRinit(model)
   for k,v in pairs(model:findModules('nn.SpatialConvolution')) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if v.bias then v.bias:zero() end
   end
end

------------------------------------------------------------------------------------------------------------

local function FCinit(model)
   for k,v in pairs(model:findModules'nn.Linear') do
     v.bias:zero()
   end
end

------------------------------------------------------------------------------------------------------------

local function DisableBias(model)
   for i,v in ipairs(model:findModules'nn.SpatialConvolution') do
      v.bias = nil
      v.gradBias = nil
   end
end

------------------------------------------------------------------------------------------------------------

local function cleanDPT(module)
   -- This assumes this DPT was created by the function above: all the
   -- module.modules are clones of the same network on different GPUs
   -- hence we only need to keep one when saving the model to the disk.
   local newDPT = nn.DataParallelTable(1, true, true)
   cutorch.setDevice(opt.GPU)
   newDPT:add(module:get(1), opt.GPU)
   return newDPT
end

------------------------------------------------------------------------------------------------------------

local function makeDataParallelTable(model, nGPU)
   if nGPU > 1 then
      local gpus = torch.range(1, nGPU):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      --local dpt = nn.DataParallelTable(1, true, true)
      local dpt = nn.DataParallelTable(1)
         :add(model, gpus)
         :threads(function()
            require 'nngraph'
            paths.dofile('../modules/NoBackprop.lua')
            if pcall(require,'cudnn') then
               local cudnn = require 'cudnn'
               cudnn.fastest, cudnn.benchmark = fastest, benchmark
            end
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end
   return model
end

------------------------------------------------------------------------------------------------------------

local function saveDataParallel(filename, model)
   if torch.type(model) == 'nn.DataParallelTable' then
      --torch.save(filename, cleanDPT(model))
      torch.save(filename, model.modules[1])
   elseif torch.type(model) == 'nn.gModule' then
      torch.save(filename, model)
   elseif torch.type(model) == 'nn.Sequential' then
      local temp_model = nn.Sequential()
      for i, module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            temp_model:add(cleanDPT(module))
         else
            temp_model:add(module)
         end
      end
      torch.save(filename, temp_model)
   else
      error('This saving function only works with Sequential or DataParallelTable modules.')
   end
end

------------------------------------------------------------------------------------------------------------

local function loadDataParallel(model, nGPU)
   --if opt.backend == 'cudnn' then
   --   require 'cudnn'
   --end
   if torch.type(model) == 'nn.DataParallelTable' then
      return makeDataParallelTable(model:get(1):float(), nGPU)
   elseif torch.type(model) == 'nn.gModule' then
      return makeDataParallelTable(model, nGPU)
   elseif torch.type(model) == 'nn.Sequential' then
      for i,module in ipairs(model.modules) do
         if torch.type(module) == 'nn.DataParallelTable' then
            model.modules[i] = makeDataParallelTable(module:get(1):float(), nGPU)
         end
      end
      return model
   else
      error('The loaded model is not a Sequential or DataParallelTable module.')
   end
end

------------------------------------------------------------------------------------------------------------

local function ReplicateTensor2Table(tensor, N)
    local data = {}
    if N > 1 then
        for i=1, N do table.insert(data, tensor) end
        return data
    else
        return tensor
    end
end

------------------------------------------------------------------------------------------------------------

local function Str2TableFn(input) -- convert a string into a table
    --local json = require 'rapidjson'
    local json = require 'json'

    -- replace '{' and '}' by '[' and '], respectively
    input = input:gsub("%{","[")
    input = input:gsub("%}","]")

    -- use json decode function to convert the string into a table
    return json.decode(input)
end

------------------------------------------------------------------------------------------------------------

local function ConcatTables(tableA, tableB) -- convert a string into a table
    local tableOut = {}
    for i=1, #tableA do
        table.insert(tableOut, tableA[i])
    end
    for i=1, #tableB do
        table.insert(tableOut, tableB[i])
    end
    return tableOut
end

------------------------------------------------------------------------------------------------------------

local function ConvertString2Boolean(var) -- converts string to booleans
    if type(var) == 'string' then
        local str = string.lower(var):gsub("%s+", "")
        str = string.gsub(str, "%s+", "")
        if str == 'true' then
            return true
        elseif str == 'false' then
            return false
        else
            error('Cannot convert input to boolean type: ' .. var)
        end
    elseif type(var) == 'boolean' then
        return var
    else
        error('Input variable is not of string/boolean type: ' .. type(var))
    end
end

------------------------------------------------------------------------------------------------------------

return {
   MSRinit = MSRinit,
   FCinit = FCinit,
   DisableBias = DisableBias,

   makeDataParallelTable = makeDataParallelTable,
   saveDataParallel = saveDataParallel,
   loadDataParallel = loadDataParallel,

   -- table operations
   ReplicateTensor2Table = ReplicateTensor2Table,
   Str2TableFn = Str2TableFn,
   ConcatTables = ConcatTables,
   Str2Bool = ConvertString2Boolean,
}