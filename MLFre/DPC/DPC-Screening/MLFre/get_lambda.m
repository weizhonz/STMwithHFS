function lambda = get_lambda( lb, ub, npar, scale )
%% this function set the parameter sequence
%  lb: the lower bound of the parameter sequence
%  ub: the upper bound of the parameter sequence
%  npar: the number of parameters 
%  scale: can be 'linear' or 'log'
if strcmp(scale, 'linear')
    delta_lambda = (ub - lb)/(npar-1);
    lambda=lb:delta_lambda:ub;
elseif strcmp(scale, 'log')
    delta_lambda = (log(ub) - log(lb))/(npar-1);
    lambda = exp(log(lb):delta_lambda:log(ub));
else
    error('ErrorTests:convertTest', 'The scale of the parameter sequence can be linear or logrithmic.\nPlease check your input and try again.');
end
end

