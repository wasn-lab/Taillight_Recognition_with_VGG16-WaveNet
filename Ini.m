xy_ends_Pos = Map_data(:,1:5);
nodes = zeros(length(xy_ends_Pos(:,1))*2,3);
segments = zeros(length(xy_ends_Pos(:,1)),3);
segments(:,1) = xy_ends_Pos(:,1);

for i = 1 : length(xy_ends_Pos(:,1))
    if xy_ends_Pos(i,2) == 0 && xy_ends_Pos(i,3) == 0
        index_node = 1;
        nodes(index_node,1) = index_node;
        nodes(index_node,2:3) = xy_ends_Pos(i,2:3);
        break;
    elseif xy_ends_Pos(i,4) == 0 && xy_ends_Pos(i,5) == 0
        index_node = 1;
        nodes(index_node,1) = index_node;
        nodes(index_node,2:3) = xy_ends_Pos(i,4:5);
        break;
    else
        index_node = 0;
    end
end

for i = 1 : length(xy_ends_Pos(:,1))
    [num,index] = min((nodes(:,2)-xy_ends_Pos(i,2)).^2+(nodes(:,3)-xy_ends_Pos(i,3)).^2);
    if num ~= 0
        index_node = index_node + 1;
        nodes(index_node,1) = index_node;
        nodes(index_node,2:3) = xy_ends_Pos(i,2:3);
    end
    [num,index] = min((nodes(:,2)-xy_ends_Pos(i,4)).^2+(nodes(:,3)-xy_ends_Pos(i,5)).^2);
    if num ~= 0
        index_node = index_node + 1;
        nodes(index_node,1) = index_node;
        nodes(index_node,2:3) = xy_ends_Pos(i,4:5);
    end
end
nodes = nodes(1:index_node,:);

for i = 1 : length(xy_ends_Pos(:,1))
    Ini_x_node_num = find(nodes(:,2) == xy_ends_Pos(i,2));
    Ini_y_node_num = find(nodes(:,3) == xy_ends_Pos(i,3));
    if length(Ini_x_node_num) == 1
        segments(i,2) = nodes(Ini_x_node_num,1);
    else
        for j = 1 : length(Ini_y_node_num(:))
            temp = find(Ini_x_node_num(:) == Ini_y_node_num(j));
            if isempty(temp) ~= 1
                break;
            end
        end
            segments(i,2) = nodes(Ini_x_node_num(temp),1);
    end
    
    End_x_node_num = find(nodes(:,2) == xy_ends_Pos(i,4));
    End_y_node_num = find(nodes(:,3) == xy_ends_Pos(i,5));
    if length(End_x_node_num) == 1
        segments(i,3) = nodes(End_x_node_num,1);
    else
        for j = 1 : length(End_y_node_num(:))
            temp = find(End_x_node_num(:) == End_y_node_num(j));
            if isempty(temp) ~= 1
                break;
            end
        end
            segments(i,3) = nodes(End_x_node_num(temp),1);
    end
end

%% Main Static route planning
start_node_id = 1;
finish_node_id = length(Map_data(:,1));%38;

%% System setting
delta_t = 0.01;

rmin = 10;
% Veh_Velocity = 5;
forward_length = 15;
Veh_size = [1.99986,6.09558]; % [Width , Length], unit m
Veh_CG = [0.5, 0.2];  % [Width portion from left side, Length portion from front side]

% velocity
ay_max = 2;%%%%%%%%%%%%%%%%%%%
vmax = 6;%35/3.6;
k_safe = 1;%%%%%%%%%%%%%%

%% obstcale bunding box position (Input)
OB_num = 1;%length(OBXY(:,1))/2;

%% (Input) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end_right_pass = 0; % can pass 1
end_left_pass = 1; % can't pass 0