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

function show_images(in_image, out_image)
    screen_size = get(0, 'ScreenSize');
    image_size = size(in_image);
    w_scr = screen_size(3);
    h_scr = screen_size(4);
    w_fig = w_scr * 0.4;
    h_fig = w_fig * image_size(1) / image_size(2);
    w_dist = w_scr * 0.02;
    in_pos = [w_scr / 2 - w_fig - w_dist / 2, h_scr - h_fig, w_fig, h_fig];
    out_pos = in_pos;
    out_pos(1) = w_scr / 2 + w_dist / 2;
    
    in_fig = figure('Name','Input','Visible','Off');
    imshow(in_image);
    set(in_fig, 'Position', in_pos, 'Visible', 'On');
    
    out_fig = figure('Name','Result','Visible','Off');
    imshow(out_image);
    set(out_fig, 'Position', out_pos, 'Visible', 'On');
end
