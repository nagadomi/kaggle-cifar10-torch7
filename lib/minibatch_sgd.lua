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
   for t = 1, train_x:size(1), batch_size do
      if t + batch_size > train_x:size(1) then
	 break
      end
      xlua.progress(t, train_x:size(1))
      local inputs = torch.Tensor(batch_size,
				  train_x:size(2),
				  train_x:size(3),
				  train_x:size(4))
      local targets = torch.Tensor(batch_size,
				   train_y:size(2))
      for i = 1, batch_size do
         local input = train_x[shuffle[t + i - 1]]
         local target = train_y[shuffle[t + i - 1]]
         inputs[i]:copy(input)
	 targets[i]:copy(target)
      end
      inputs = inputs:cuda()
      targets = targets:cuda()
      
      local feval = function(x)
	 if x ~= parameters then
	    parameters:copy(x)
	 end
	 gradParameters:zero()
	 local f = 0
	 for i = 1, inputs:size(1) do
	    local output = model:forward(inputs[i])
	    local err = criterion:forward(output, targets[i])
	    f = f + err
	    confusion:add(output, targets[i])
	    local df_do = criterion:backward(output, targets[i])
	    model:backward(inputs[i], df_do)
	 end
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
