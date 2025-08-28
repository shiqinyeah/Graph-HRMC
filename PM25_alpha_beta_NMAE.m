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
% ��ǰȱʧ��
currentRate = 0.5;
% ���㱾����Ҫ����ȱʧ��Ԫ����
totalMissing = round(currentRate * numel(GroundTruth));
additionalMissing = totalMissing - nnz(missingMatrix);
% ���ѡ������ȱʧ��λ��
availableIndices = find(~missingMatrix);
newMissingIndices = availableIndices(randperm(length(availableIndices), additionalMissing));
% ����ȱʧ����
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

%% ������Сֵλ��
[minValue, linearIndex] = min(NMAE_our(:)); 
[row, col] = ind2sub(size(NMAE_our), linearIndex);
J=(row-1)*6+[1,2,3,4];

%% ��ά��״ͼ
figure;
b = bar3(NMAE_our);
zlim([min(NMAE_our(:)) * 0.99, max(NMAE_our(:)) * 1.01]);

% ����������
set(gca,'LineWidth',1,'FontSize',16,'FontName','Helvetica')
set(gca,'XTickLabel',alpha2,'YTickLabel',beta2)

% ��ǩ�ͱ���
hXLabel = xlabel('\alpha_{2}');
hYLabel = ylabel('\beta_{2}');
hZLabel = zlabel('NMAE');
set([hXLabel,hYLabel,hZLabel],'FontSize',16,'FontName','Helvetica');
title('PM 2.5','FontSize',20,'FontName','Helvetica');

% �ҵ���Сֵ��λ��
[min_val, min_idx] = min(NMAE_our(:));
[row_min, col_min] = ind2sub(size(NMAE_our), min_idx);

% ������״ͼ��ɫ
for i = 1:length(b)
    zdata = b(i).ZData; % ��ȡ���ӵ� Z ����
    cdata = zeros(size(zdata)); % ��ʼ����ɫ����

    % ����ÿ�����ӵ����ж�
    for j = 1:size(zdata, 1)
        if i == col_min && ismember(j, J) % ��Сֵ����
            cdata(j, :) = 2; % ����ֵ2���Ӧ��map2�е���ɫ
        else
            cdata(j, :) = 1; % ��������: ����ֵ1���Ӧ��map1�е���ɫ
        end
    end

    % Ӧ����ɫ����
    b(i).CData = cdata; % ������ɫ����
end

map1 = addcolorplus(203); 
map2 = addcolorplus(180);
colormap([map1;map2]); % �Զ�����ɫӳ��