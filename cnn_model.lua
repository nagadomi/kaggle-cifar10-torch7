require 'cunn'

-- Tranditional CNN model
function cnn_model() -- validate.lua Acc: 0.88
   local model = nn.Sequential() 
   
   -- convolution layers
   model:add(nn.SpatialConvolutionMM(3, 128, 5, 5, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   
   model:add(nn.SpatialConvolutionMM(128, 256, 5, 5, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   
   model:add(nn.SpatialZeroPadding(1, 1, 1, 1))
   model:add(nn.SpatialConvolutionMM(256, 512, 4, 4, 1, 1))
   model:add(nn.ReLU())
   
   -- fully connected layers
   model:add(nn.SpatialConvolutionMM(512, 1024, 2, 2, 1, 1))
   model:add(nn.ReLU())
   model:add(nn.Dropout(0.5))
   model:add(nn.SpatialConvolutionMM(1024, 10, 1, 1, 1, 1))
   
   model:add(nn.Reshape(10))
   model:add(nn.SoftMax())
   
   return model
end
