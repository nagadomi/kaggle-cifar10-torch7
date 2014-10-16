require 'torch'
require 'image'
require 'sys'

-- memory efficient version of unsup.zca_whiten/unsup.pcacov.
-- original version can be found at: https://github.com/koraykv/unsup/
local function pcacov(x, means)
   for i = 1, x:size(1) do
      x[i]:add(-1, means)
   end
   local c = torch.mm(x:t(), x)
   for i = 1, x:size(1) do
      x[i]:add(means)
   end
   c:div(x:size(1)-1)
   local ce,cv = torch.symeig(c,'V')
   return ce,cv
end
local function zca_whiten(data, means, P, invP, epsilon)
    local epsilon = epsilon or 1e-5
    local data_size = data:size()
    local dims = data:size()
    local nsamples = dims[1]
    local n_dimensions = data:nElement() / nsamples
    if data:dim() >= 3 then
       data = data:view(nsamples, n_dimensions)
    end
    if not means or not P or not invP then 
        -- compute mean vector if not provided 
       means = torch.mean(data, 1)
        -- compute transformation matrix P if not provided
       local ce, cv = pcacov(data, means)
       collectgarbage()
       ce:add(epsilon):sqrt()
       local invce = ce:clone():pow(-1)
       local invdiag = torch.diag(invce)
       P = torch.mm(cv, invdiag)
       P = torch.mm(P, cv:t())

        -- compute inverse of the transformation
       local diag = torch.diag(ce)
       invP = torch.mm(cv, diag)
       invP = torch.mm(invP, cv:t())
    end
    collectgarbage()
    -- remove the means
    for i = 1, data:size(1) do
       data[i]:add(-1, means)
    end
    -- transform in ZCA space
    if data:size(1) > 100000 then
       -- matrix mul with 16-spliting
       local step = math.floor(data:size(1) / 16)
       for i = 1, data:size(1), step do
	  local n = step
	  if i + n > data:size(1) then
	     n = data:size(1) - i
	  end
	  if n > 0 then
	     data:narrow(1, i, n):copy(torch.mm(data:narrow(1, i, n), P))
	  end
	  collectgarbage()
       end
    else
       data:copy(torch.mm(data, P))
    end
    data = data:view(data_size)
    collectgarbage()
    
    return data, means, P, invP
end
local function zca(x, means, P, invP)
   local ax
   ax, means, P, invP = zca_whiten(x, means, P, invP, 0.01) -- 0.1
   x:copy(ax)
   return means, P, invP
end

local function global_contrast_normalization(x, mean, std)
   local scale = 1.0
   local u = mean or x:mean(1)
   local v = std or (x:std(1):div(scale))
   for i = 1, x:size(1) do
      x[i]:add(-u)
      x[i]:cdiv(v)
   end
   return u, v
end
function preprocessing(x, params)
   params = params or {}
   params['gcn_mean'], params['gcn_std'] = global_contrast_normalization(x, params['gcn_mean'], params['gcn_std'])
   params['zca_u'], params['zca_p'], params['zca_invp'] = zca(x, params['zca_u'], params['zca_p'], params['zca_invp'])
   
   return params
end

return true
