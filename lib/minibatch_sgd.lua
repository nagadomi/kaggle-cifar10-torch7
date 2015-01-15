require 'optim'
require 'xlua'

function minibatch_sgd(model, criterion,
		       train_x, train_y,
		       classes, config)
   local parameters, gradParameters = model:getParameters()
   local confusion = optim.ConfusionMatrix(classes)
   config = config or {}
   local batch_size = config.xBatchSize or 12
   local shuffle = torch.randperm(train_x:size(1))
   local c = 1
   local inputs = torch.CudaTensor(batch_size,
				   train_x:size(2),
				   train_x:size(3),
				   train_x:size(4))
   local targets = torch.CudaTensor(batch_size,
				    train_y:size(2))
   for t = 1, train_x:size(1), batch_size do
      if t + batch_size > train_x:size(1) then
	 break
      end
      xlua.progress(t, train_x:size(1))
      for i = 1, batch_size do
         inputs[i]:copy(train_x[shuffle[t + i - 1]])
	 targets[i]:copy(train_y[shuffle[t + i - 1]])
      end
      
      local feval = function(x)
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 gradParameters:zero()
	 local f = 0
	 local output = model:forward(inputs)
	 local df_do = torch.Tensor(output:size(1), targets:size(2))
	 for k = 1, output:size(1) do
	    local err = criterion:forward(output[k], targets[k])
	    f = f + err
	    df_do[k]:copy(criterion:backward(output[k], targets[k]))
	    confusion:add(output[k], targets[k])
	 end
	 model:backward(inputs, df_do:cuda())
	 gradParameters:div(inputs:size(1))
	 f = f / inputs:size(1)
	 return f, gradParameters
      end
      optim.sgd(feval, parameters, config)
      
      c = c + 1
      if c % 1000 == 0 then
	 collectgarbage("collect")
      end
   end
   xlua.progress(train_x:size(1), train_x:size(1))
   
   return confusion
end
return true
