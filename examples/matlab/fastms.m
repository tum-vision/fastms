%
% This file is part of fastms.
% 
% Copyright 2014 Evgeny Strekalovskiy <evgeny dot strekalovskiy at in dot tum dot de> (Technical University of Munich)
% 
% fastms is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% fastms is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with fastms. If not, see <http://www.gnu.org/licenses/>.
% 

function [out_image] = fastms(in_image, varargin)

    % TODO: maybe use inputParser of matlab
    options = struct(...
        'lambda', [], ...
		'alpha', [], ...
		'temporal', [], ...
		'iterations', [], ...
		'stop_eps', [], ...
		'stop_k', [], ...
		'adapt_params', [], ...
		'weight', [], ...
		'use_double', [], ...
		'engine', [], ...
		'edges', [], ...
		'verbose', []);

    option_names = fieldnames(options);
    num_args = length(varargin);
    if rem(num_args,2) ~= 0
        error('Options must be name/value pairs');
    end
    for pair = reshape(varargin,2,[]) % pair = {name;value}
        in_name = lower(pair{1});
        if any(strmatch(in_name, option_names))
            options.(in_name) = pair{2};
        else
            error('Unexpected parameter name %s', in_name);
        end
    end

    out_image = fastms_mex(in_image, options);
end
