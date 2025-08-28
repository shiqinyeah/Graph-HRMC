clc;clear;
load('PM25_tensor.mat')
T=double(Y_source);
[s1, s2, s3] = size(T);
GroundTruth=[];
for k= 1:s3
    slice = squeeze(T(:,:,k));
    GroundTruth=[GroundTruth slice];
end
GroundTruth=GroundTruth(:,1:300);

%% Normalization
min_value = min(GroundTruth(:));
max_value = max(GroundTruth(:));
GroundTruth = (GroundTruth - min_value) / (max_value - min_value);
[M,N]=size(GroundTruth);

%% sigma
Dist=pdist2(GroundTruth',GroundTruth','euclidean');
Dmax=max(Dist(:));
sigma=Dmax;

%% parameter
alpha2=[0.001,0.005,0.01,0.05,0.1,0.5,1];  % 0.05
LENGTH1=length(alpha2);
beta2=[0.001,0.005,0.01,0.05,0.1,0.5,1];  % 0.001
LENGTH2=length(beta2);

parameter.RANK=10;
parameter.sigma=sigma;
parameter.threshold=1e-5;
parameter.IterMax=200;

%% Boolean
missingMatrix = false(size(GroundTruth));
% 当前缺失率
currentRate = 0.5;
% 计算本次需要新增缺失的元素数
totalMissing = round(currentRate * numel(GroundTruth));
additionalMissing = totalMissing - nnz(missingMatrix);
% 随机选择新增缺失的位置
availableIndices = find(~missingMatrix);
newMissingIndices = availableIndices(randperm(length(availableIndices), additionalMissing));
% 更新缺失矩阵
missingMatrix(newMissingIndices) = true;
missingMatrix=double(missingMatrix);
Boolean=1-missingMatrix;

%% Exam
NMAE_our=zeros(LENGTH2,LENGTH1);
for i=1:LENGTH2
    parameter.beta2=beta2(i);
    for j=1:LENGTH1
        parameter.alpha2=alpha2(j);
        Xour=Graph_HRMC_ab12(GroundTruth,Boolean,parameter);
        err=Xour-GroundTruth;
        NMAE_our(i,j)=sum(abs(err(:)))/sum(abs(GroundTruth(:)));
    end
end

%% 查找最小值位置
[minValue, linearIndex] = min(NMAE_our(:)); 
[row, col] = ind2sub(size(NMAE_our), linearIndex);
J=(row-1)*6+[1,2,3,4];

%% 三维柱状图
figure;
b = bar3(NMAE_our);
zlim([min(NMAE_our(:)) * 0.99, max(NMAE_our(:)) * 1.01]);

% 坐标轴设置
set(gca,'LineWidth',1,'FontSize',16,'FontName','Helvetica')
set(gca,'XTickLabel',alpha2,'YTickLabel',beta2)

% 标签和标题
hXLabel = xlabel('\alpha_{2}');
hYLabel = ylabel('\beta_{2}');
hZLabel = zlabel('NMAE');
set([hXLabel,hYLabel,hZLabel],'FontSize',16,'FontName','Helvetica');
title('PM 2.5','FontSize',20,'FontName','Helvetica');

% 找到最小值的位置
[min_val, min_idx] = min(NMAE_our(:));
[row_min, col_min] = ind2sub(size(NMAE_our), min_idx);

% 设置柱状图颜色
for i = 1:length(b)
    zdata = b(i).ZData; % 获取柱子的 Z 数据
    cdata = zeros(size(zdata)); % 初始化颜色数据

    % 遍历每根柱子的所有段
    for j = 1:size(zdata, 1)
        if i == col_min && ismember(j, J) % 最小值柱子
            cdata(j, :) = 2; % 索引值2会对应到map2中的颜色
        else
            cdata(j, :) = 1; % 其他柱子: 索引值1会对应到map1中的颜色
        end
    end

    % 应用颜色数据
    b(i).CData = cdata; % 设置颜色数据
end

map1 = addcolorplus(203); 
map2 = addcolorplus(180);
colormap([map1;map2]); % 自定义颜色映射