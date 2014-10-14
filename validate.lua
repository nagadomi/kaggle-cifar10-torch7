require 'cutorch'
require 'cunn'
require './SETTINGS'
require './lib/minibatch_sgd'
require './lib/data_augmentation'
require './lib/preprocessing'
require './very_deep_model.lua'

local function test(model, params, test_x, test_y, classes)
   local confusion = optim.ConfusionMatrix(classes)
   for i = 1, test_x:size(1) do
      local preds = torch.Tensor(10):zero()
      local x = data_augmentation(test_x[i])
      local step = 64
      preprocessing(x, params)
      for j = 1, x:size(1), step do
	 local batch = torch.Tensor(step, x:size(2), x:size(3), x:size(4)):zero()
	 local n = step
	 if j + n > x:size(1) then
	    n = 1 + n - ((j + n) - x:size(1))
	 end
	 batch:narrow(1, 1, n):copy(x:narrow(1, j, n))
	 local z = model:forward(batch:cuda()):float()
	 -- averaging
	 for k = 1, n do
	    preds = preds + z[k]
	 end
      end
      preds:div(x:size(1))
      confusion:add(preds, test_y[i])
      xlua.progress(i, test_x:size(1))
   end
   xlua.progress(test_x:size(1), test_x:size(1))
   return confusion
end

local function validation()
   local TRAIN_SIZE = 40000
   local TEST_SIZE = 10000
   local MAX_EPOCH = 20

   local x = torch.load(string.format("%s/train_x.bin", DATA_DIR))
   local y = torch.load(string.format("%s/train_y.bin", DATA_DIR))
   local train_x = x:narrow(1, 1, TRAIN_SIZE)
   local train_y = y:narrow(1, 1, TRAIN_SIZE)
   local test_x = x:narrow(1, TRAIN_SIZE + 1, TEST_SIZE)
   local test_y = y:narrow(1, TRAIN_SIZE + 1, TEST_SIZE)
   local model = very_deep_model():cuda()
   local criterion = nn.MSECriterion():cuda()
   local sgd_config = {
      learningRate = 1.0,
      learningRateDecay = 5.0e-6,
      momentum = 0.9,
      xBatchSize = 64
   }
   local params = nil
   
   print("data augmentation ..")
   train_x, train_y = data_augmentation(train_x, train_y)
   collectgarbage()
   
   print("preprocessing ..")
   params = preprocessing(train_x)
   collectgarbage()
   
   for epoch = 1, MAX_EPOCH do
      if epoch == MAX_EPOCH then
	 -- final epoch
	 sgd_config.learningRateDecay = 0
	 sgd_config.learningRate = 0.01
      end
      model:training()
      print("# " .. epoch)
      print("## train")
      print(minibatch_sgd(model, criterion, train_x, train_y,
			  CLASSES, sgd_config))
      print("## test")
      model:evaluate()
      print(test(model, params, test_x, test_y, CLASSES))
      
      collectgarbage()
   end
end
torch.manualSeed(11)
validation()
