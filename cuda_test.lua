require 'cutorch'
require 'cunn'
require './nin_model'

local function cuda_test()
   local model = nin_model():cuda()
   local criterion = nn.MSECriterion():cuda()
   local x = torch.Tensor(16, 3, 24, 24):uniform():cuda()
   local y = torch.Tensor(16, 10):bernoulli(0.1):cuda()
   for i = 1, 16 do
      local z = model:forward(x[i])
      local err = criterion:forward(z, y[i])
      local df_do = criterion:backward(z, y[i])
      model:backward(x[i], df_do)
   end
   print("CUDA Test Successful!")
end

torch.setdefaulttensortype('torch.FloatTensor')
print(cutorch.getDeviceProperties(cutorch.getDevice()))
cuda_test()
