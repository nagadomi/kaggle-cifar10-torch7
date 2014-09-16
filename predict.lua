require 'cutorch'
require 'cunn'
require './SETTINGS'
require './lib/data_augmentation'
require './lib/preprocessing'
require './nin_model.lua'

local function predict(file, model, params, test_x)
   local fp = io.open(file, "w")
   fp:write("id,label\n")
   for i = 1, test_x:size(1) do
      local preds = torch.Tensor(10):zero()
      local x = data_augmentation(test_x[i])
      local batch = torch.Tensor(64, x:size(2), x:size(3), x:size(4)):zero()
      preprocessing(x, params)
      batch:narrow(1, 1, x:size(1)):copy(x)
      local z = model:forward(batch:cuda()):float()
      for j = 1, x:size(1) do
	 preds = preds + z[j]
      end
      preds:div(x:size(1))
      
      local max_v, max_i = preds:max(1)
      fp:write(string.format("%d,%s\n", i, ID2LABEL[max_i[1]]))
      xlua.progress(i, test_x:size(1))
      if i % 1000 == 0 then
	 collectgarbage()
      end
   end
   xlua.progress(test_x:size(1), test_x:size(1))
   fp:close()
end
local function prediction()
   local x = torch.load(string.format("%s/test_x.bin", DATA_DIR))
   local model = torch.load("models/very_deep_15.model"):cuda()
   local params = torch.load("models/preprocessing_params.bin")
   
   predict("./submission.txt", model, params, x)
end

prediction()
