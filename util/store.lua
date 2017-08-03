--[[
    Model storing functions to disk.
]]


if not utils then utils = paths.dofile('utils.lua') end

------------------------------------------------------------------------------------------------------------

local function store(model, optimState, epoch, opt, flag)
    local filename_model, filename_optimstate
    if flag then
        filename_model = paths.concat(opt.save,'model_' .. epoch ..'.t7')
        filename_optimstate = paths.concat(opt.save,'optim_' .. epoch ..'.t7')
    else
        filename_model = paths.concat(opt.save,'model.t7')
        filename_optimstate = paths.concat(opt.save,'optim.t7')
    end

    print('Saving model snapshot to: ' .. filename_model)
    torch.save(filename_optimstate, optimState)
    torch.save(paths.concat(opt.save,'last_epoch.t7'), epoch)
    if opt.clear_buffers then
        torch.save(filename_model, model:clearState())
    else
        torch.save(filename_model, model)
    end

    -- make a symlink to the last trained model
    local filename_symlink = paths.concat(opt.save,'model_final.t7')
    if paths.filep(filename_symlink) then
        os.execute(('rm %s'):format(filename_symlink))
    end
    os.execute(('ln -s %s %s'):format(filename_model, filename_symlink))
end

------------------------------------------------------------------------------------------------------------

function storeModel(model, optimState, epoch, opt)
--[[ store model snapshot ]]

    if opt.snapshot > 0 then
        if epoch%opt.snapshot == 0 then
           store(model, optimState, epoch, opt, true)
        end
    elseif opt.snapshot < 0 then
        if epoch%math.abs(opt.snapshot) == 0 then
           store(model, optimState, epoch, opt, false)
        end
    else
        -- save only at the last epoch
        if epoch == opt.nEpochs then
          store(model, optimState, epoch, opt, false)
        end
    end
end

------------------------------------------------------------------------------------------------------------

function storeModelBest(model, opt)
--[[ store model snapshot of the best model]]

    local filename_model = paths.concat(opt.save, 'best_model_accuracy.t7')

    print('New best accuracy detected! Saving model snapshot to disk: ' .. filename_model)

    if opt.clear_buffers then
        torch.save(filename_model, model:clearState())
    else
        torch.save(filename_model, model)
    end
end