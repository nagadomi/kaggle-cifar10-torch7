require 'torch'
require 'image'

function save_images(x, n, file)
   file = file or "./out.png"
   local input = x:narrow(1, 1, n)
   local view = image.toDisplayTensor({input = input,
				       padding = 2,
				       nrow = 9,
				       symmetric = true})
   image.save(file, view)
end

return true