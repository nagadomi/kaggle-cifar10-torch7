require 'cutorch'
require './SETTINGS'
require './lib/minibatch_sgd'
require './lib/data_augmentation'
require './lib/preprocessing'
require './nin_model.lua'

function training()
   local MAX_EPOCH = 10
   local x = torch.load(string.format("%s/train_x.bin", DATA_DIR))
   local y = torch.load(string.format("%s/train_y.bin", DATA_DIR))
   local model = nin_model():cuda()
   local criterion = nn.MSECriterion():cuda()
   local sgd_config = {
      learningRate = 0.1,
      learningRateDecay = 5.0e-6,
      momentum = 0.9,
      xBatchSize = 12
   }
   local params = nil

   print("data augmentation ..")
   x, y = data_augmentation(x, y)
   collectgarbage()
   
   print("preprocessing ..")
   params = preprocessing(x)
   torch.save("models/preprocessing_params.bin", params)
   collectgarbage()
   
   for epoch = 1, MAX_EPOCH do
      print("# " .. epoch)
      if epoch == MAX_EPOCH then
	 -- final epoch
	 sgd_config.learningRateDecay = 0
	 sgd_config.learningRate = 0.001
      end
      model:training()
      print(minibatch_sgd(model, criterion, x, y,
			  CLASSES, sgd_config))
      model:evaluate()
      torch.save(string.format("models/nin_%d.model", epoch), model)
      epoch = epoch + 1
      
      collectgarbage()
   end
end
torch.manualSeed(11)
training()
