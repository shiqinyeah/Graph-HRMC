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
RANK=1:20;  LENGTH=length(RANK);
parameter.sigma=sigma;
parameter.threshold=1e-6;
parameter.alpha2=0.01;
parameter.beta2=0.01;
parameter.IterMax=200;

%% miss parameter
missRates = 0.5:0.1:0.8;
missLength=length(missRates);
RMSE_our=zeros(missLength,LENGTH);
NMAE_our=zeros(missLength,LENGTH);
missingMatrix = false(size(GroundTruth));

%% Exam
for i = 1:missLength
    % ��ǰȱʧ��
    currentRate = missRates(i);

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

    for j=1:LENGTH
        parameter.RANK=RANK(j);
        Xour=Graph_HRMC_ab12(GroundTruth,Boolean,parameter);
        err_our=Xour-GroundTruth;
        RMSE_our(i,j)=norm(err_our,'fro')/norm(GroundTruth,'fro');
        NMAE_our(i,j)=sum(abs(err_our(:)))/sum(abs(GroundTruth(:)));
    end
end

%% legend
legendStr=cell(1,missLength);
parfor i = 1:missLength
    legendStr{1,i}=['Missing Rate =', num2str(missRates(i))];
end

figure(1);
plot(RANK,RMSE_our,'-o', 'LineWidth', 2); %1
set(gca, 'LineWidth', 1, 'FontSize', 16); %������������� & �����������С
xlabel('Latent Feature Rank','FontSize',20);
ylabel('RMSE','FontSize',20);
title('PM 2.5','FontSize',20);
legend(legendStr,'FontSize',14);
grid on;

figure(2);
plot(RANK,NMAE_our,'-o', 'LineWidth', 2); %1
set(gca, 'LineWidth', 1, 'FontSize', 16); %������������� & �����������С
xlabel('Latent Feature Rank','FontSize',20);
ylabel('NMAE','FontSize',20);
title('PM 2.5','FontSize',20);
legend(legendStr,'FontSize',14);
grid on;