require 'torch'
require 'image'

function save_images(x, n, file)
   file = file or "./out.png"
   local input = x:narrow(1, 1, n)
   local view = image.toDisplayTensor({input = input,
				       padding = 2,
				       nrow = math.floor(math.sqrt(input:size(1))),
				       symmetric = true})
   image.save(file, view)
end

return true