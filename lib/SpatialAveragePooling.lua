local SpatialAveragePooling, Parent = torch.class('nn.SpatialAveragePooling', 'nn.SpatialSubSampling')

function SpatialAveragePooling:__init(nInputPlane, kW, kH, dW, dH)
   Parent.__init(self, nInputPlane, kW, kH, dW, dH)
end

function SpatialAveragePooling:reset()
   self.weight:fill(1.0)
   self.bias:fill(0.0)
end
-- avoid parameter update
function SpatialAveragePooling:accGradParameters()
end
function SpatialAveragePooling:accUpdateGradParameters()
end
function SpatialAveragePooling:updateParameters()
end

return true

