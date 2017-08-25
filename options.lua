--[[
    Options configurations for the train script.
]]

projectDir = projectDir or './'

local function Parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text(' ---------- General options ------------------------------------')
    cmd:text()
    cmd:option('-expID',    'architecture-test', 'Experiment ID')
    cmd:option('-dataset',  'ucf_sports', 'Dataset choice: ucf_sports | ucf_101.')
    cmd:option('-data_dir',       'none', 'Path to store the dataset\'s data files.')
    cmd:option('-expDir',   projectDir .. '/exp',  'Experiments directory')
    cmd:option('-manualSeed',          2, 'Manually set RNG seed')
    cmd:option('-GPU',                 1, 'Default preferred GPU, if set to -1: no GPU')
    cmd:option('-nGPU',                1, 'Number of GPUs to use by default')
    cmd:option('-nThreads',            4, 'Number of data loading threads')
    cmd:text()
    cmd:text(' ---------- Model options --------------------------------------')
    cmd:text()
    cmd:option('-netType',  'vgg16-lstm', 'Options: vgg16-lstm | hms-lstm | vgg16-hms-lstm | vgg16-convnet | hms-convnet | vgg16-hms-convnet.')
    cmd:option('-nFeats',            256, 'Number of features of the rnn/conv layer.')
    cmd:option('-nLayers',             2, 'Number of rnn/conv layers.')
    cmd:option('-loadModel',      'none', 'Provide full path to a previously trained model')
    cmd:option('-continue',      'false', 'Pick up where an experiment left off')
    cmd:option('-branch',         'none', 'Provide a parent expID to branch off')
    cmd:option('-snapshot',            5, 'How often to take a snapshot of the model (0 = never)')
    cmd:option('-saveBest',       'true', 'Saves a snapshot of the model with the highest accuracy.')
    cmd:option('-clear_buffers',  'true', 'Empty network\'s buffers (gradInput, etc.) before saving the network to disk (if true).')
    cmd:option('-convert_cudnn',  'true', 'Convert a model to use the cudnn backend (saves memory but may slow down training the model)')
    cmd:text()
    cmd:text(' ---------- Hyperparameter options -----------------------------')
    cmd:text()
    cmd:option('-LR',               1e-4, 'Learning rate')
    cmd:option('-LRdecay',           0.0, 'Learning rate decay')
    cmd:option('-momentum',          0.0, 'Momentum')
    cmd:option('-weightDecay',       0.0, 'Weight decay')
    cmd:option('-optMethod',      'adam', 'Optimization method: rmsprop | sgd | nag | adadelta | adagrad | adam.')
    cmd:option('-threshold',        .001, 'Threshold (on validation accuracy growth) to cut off training early')
    cmd:text()
    cmd:text(' ---------- Training options -----------------------------------')
    cmd:text()
    cmd:option('-trainIters',       300, 'Number of train iterations per epoch')
    cmd:option('-testIters',        100, 'Number of test iterations per epoch')
    cmd:option('-nEpochs',           15, 'Total number of epochs to run')
    cmd:option('-seq_length',        32, 'Sequence length (number of frames per window)')
    cmd:option('-batchSize',          4, 'Mini-batch size')
    cmd:option('-grad_clip',         10, 'Gradient clipping (to prevent exploding gradients).')
    cmd:option('-dropout',          0.2, 'Dropout rate.')
    cmd:option('-verbose',       "true", 'Display text info on screen')
    cmd:text()
    cmd:text(' ---------- Data options ---------------------------------------')
    cmd:text()
    cmd:option('-inputRes',       256, 'Input image resolution')
    cmd:option('-scale',          .25, 'Degree of scale augmentation')
    cmd:option('-rotate',          30, 'Degree of rotation augmentation')
    cmd:option('-rotRate',        0.5, 'Probability to apply rotation to an image.')
    cmd:text()
    cmd:text(' ---------- Test options ---------------------------------------')
    cmd:text()
    cmd:option('-test_progressbar', 'false', 'Display progressbar instead of text.')
    cmd:option('-test_load_best',   'true', 'Load the best accuracy network instead of the last snapshot.')
    cmd:option('-predictions',     0, 'Generate a predictions file (0-false | 1-true)')
    cmd:text()
    cmd:text(' ---------- Demo options --------------------------------------')
    cmd:text()
    cmd:option('-demo_nvideos',      5, 'Number of samples to display predictions.')
    cmd:option('-demo_video_ids', '{}', 'Selects specific video ids to display (disables -demo_nvideos option).')
    cmd:option('-demo_topk',         5, 'Display the top-k activities.')
    cmd:option('-demo_plot_save', 'false', 'Save plots to disk.')
    cmd:text()


    local opt = cmd:parse(arg or {})

    if not utils then
        utils = paths.dofile('util/utils.lua')
    end

    opt.continue       = utils.Str2Bool(opt.continue)
    opt.clear_buffers  = utils.Str2Bool(opt.clear_buffers)
    opt.convert_cudnn  = utils.Str2Bool(opt.convert_cudnn)
    opt.saveBest       = utils.Str2Bool(opt.saveBest)
    opt.demo_plot_save = utils.Str2Bool(opt.demo_plot_save)
    opt.verbose        = utils.Str2Bool(opt.verbose)
    opt.test_progressbar = utils.Str2Bool(opt.test_progressbar)
    opt.test_load_best = utils.Str2Bool(opt.test_load_best)

    opt.demo_video_ids = utils.Str2TableFn(opt.demo_video_ids)

    opt.expDir = paths.concat(opt.expDir, opt.dataset)
    opt.save = paths.concat(opt.expDir, opt.expID)
    if opt.loadModel == '' or opt.loadModel == 'none' then
        if opt.test_load_best then
            opt.load = paths.concat(opt.save, 'best_model_accuracy.t7')
        else
            opt.load = paths.concat(opt.save, 'model_final.t7')
        end
    else
        opt.load = opt.loadModel
    end

    if string.lower(opt.data_dir) == 'none' then
        opt.data_dir = ''
    end

    return opt
end

---------------------------------------------------------------------------------------------------

return {
  parse = Parse
}
