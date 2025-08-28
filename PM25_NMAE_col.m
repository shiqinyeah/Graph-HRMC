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

%% Parameters
RANK=15;  
threshold=1e-5;
IterMax=200;
rk=2;

parameter.RANK=RANK;
parameter.sigma=sigma;
parameter.threshold=threshold;
parameter.IterMax=IterMax;


%% miss parameter
[m, n] = deal(M,N);  
total_elements = m * n;  
missRates = 0.1:0.1:0.8;  
missLength = length(missRates);  
current_mask = ones(m, n); % 初始化第一个缺失矩阵

%% Initialization
NMAE_our=zeros(1, missLength);
NMAE_ablation=zeros(1, missLength);
NMAE_vmc=zeros(1, missLength);
NMAE_ladmc=zeros(1, missLength);
NMAE_kfmc=zeros(1, missLength);
NMAE_fnnm=zeros(1, missLength);
NMAE_pmc = zeros(1, missLength);
NMAE_Admm= zeros(1, missLength);

addpath('C:\Users\admin\Desktop\ST-HRMC\VS\VMC\')
addpath('C:\Users\admin\Desktop\ST-HRMC\VS\LADMC-main\')
addpath('C:\Users\admin\Desktop\ST-HRMC\VS\Online_HRMC\')
addpath('C:\Users\admin\Desktop\ST-HRMC\VS\PMC-AAAI-2020\')

for iter=1:missLength
    
    % 当前目标缺失率
    target_missing_rate = missRates(iter);
    num_missing = round(target_missing_rate * total_elements);  % 目标缺失元素数
    
    % 当前矩阵已有的缺失数
    current_missing = nnz(current_mask == 0);
    
    % 需要新增的缺失数
    new_missing = num_missing - current_missing;
    
    % 随机生成缺失元素位置，确保缺失位置在列中连续
    while new_missing > 0
        % 随机选择列
        rand_col = randi(n);
        
        % 当前列已缺失的元素数
        existing_missing_in_col = nnz(current_mask(:, rand_col) == 0);
        
        % 当前列剩余未缺失的元素数
        remaining_in_col = m - existing_missing_in_col;
        
        % 如果当前列还有空间缺失
        if remaining_in_col > 0
            % 随机选择一个连续的起始位置
            start_row = randi([1, remaining_in_col]);
            
            % 随机生成连续缺失的长度
            max_missing_in_col = min(new_missing, remaining_in_col);
            missing_length = randi([1, max_missing_in_col]);
            
            % 确定缺失的行索引
            end_row = min(start_row + missing_length - 1, m);
            missing_rows = start_row:end_row;
            
            % 临时矩阵用于检查全零行或全零列
            temp_mask = current_mask;
            temp_mask(missing_rows, rand_col) = 0;
            
            % 检查是否存在全零行或全零列
            if all(any(temp_mask, 2)) && all(any(temp_mask, 1))
                % 更新当前矩阵
                current_mask = temp_mask;
                new_missing = new_missing - length(missing_rows);  % 更新剩余缺失数
            end
        end
    end
    Boolean = current_mask;
    
    %% Boolean derivative
    Xinit=GroundTruth.*Boolean;
    sampmask=logical(Boolean);
    samples=GroundTruth(sampmask);

    %% 0: ablation 消融实验
    parameter.alpha2=0;
    parameter.beta2=0;
    Xablation=Graph_HRMC_ab12(GroundTruth,Boolean,parameter);
    err_ablation=Xablation-GroundTruth;
    NMAE_ablation(iter)=sum(abs(err_ablation(:)))/sum(abs(GroundTruth(:)));

    %% 1: Proposed
    alpha2=0.05;
    beta2=0.01;
    parameter.alpha2=alpha2;
    parameter.beta2=beta2;
    Xour=Graph_HRMC_ab12(GroundTruth,Boolean,parameter);
    err_our=Xour-GroundTruth;
    NMAE_our(iter)=sum(abs(err_our(:)))/sum(abs(GroundTruth(:)));

    %% 2: VMC
    options = [];
    [Xvmc,~,~,~] = vmc(Xinit,sampmask,samples,options,GroundTruth);
    err_vmc=Xvmc-GroundTruth;
    NMAE_vmc(iter)=sum(abs(err_vmc(:)))/sum(abs(GroundTruth(:)));

    %% 3: LADMC  
    Rank=nchoosek(rk+1,2);  
    Xladmc = ladmc2(Xinit,sampmask,samples,Rank,IterMax);
    err_ladmc=Xladmc-GroundTruth;
    NMAE_ladmc(iter)=sum(abs(err_ladmc(:)))/sum(abs(GroundTruth(:)));

    %% 4: KFMC (CVPR_2019)
    alpha=0.01;  beta=0.001;
    ker.type='rbf'; ker.par=[];
    options=[];
    [Xkfmc,~,~,~,~]=KFMC(Xinit,Boolean,RANK,alpha,beta,ker,options,sigma);
    err_kfmc=Xkfmc-GroundTruth;
    NMAE_kfmc(iter)=sum(abs(err_kfmc(:)))/sum(abs(GroundTruth(:)));

    %% 6: LRMC  from (CVPR_2019)     % NMF
    Xfnnm=LRMC_fnnm(Xinit,Boolean,rk,0.1);
    err_fnnm=Xfnnm-GroundTruth;
    NMAE_fnnm(iter)=sum(abs(err_fnnm(:)))/sum(abs(GroundTruth(:)));

    %% 9: PMC-W  
    w=[];
    ker.type='rbf';ker.par=[];ker.c=3;
    Xpmc_w=PMC_W(Xinit,Boolean,0.5,w,ker,IterMax,sigma);
    err_pmc_w=Xpmc_w-GroundTruth;
    NMAE_pmc(iter)=sum(abs(err_pmc_w(:)))/sum(abs(GroundTruth(:)));

    %% 10: LRMC [2017 ICML]  % SVT  根据秩截断的SVD
    options.lambda = 1e5;
    options.mu = 10;
    options.rk=rk;
    options.niter = IterMax;
    XAdmm = lrmc_admm(Xinit,sampmask,samples,options);
    err_Admm=XAdmm-GroundTruth;
    NMAE_Admm(iter)=sum(abs(err_Admm(:)))/sum(abs(GroundTruth(:)));
    
end


figure;
hold on;
plot(missRates,NMAE_fnnm,'-o', 'LineWidth', 2);  %6 %NMF
plot(missRates,NMAE_Admm,'-o', 'LineWidth', 2);  %10 %SVT
plot(missRates,NMAE_vmc,'-o', 'LineWidth', 2);   %2
plot(missRates,NMAE_ladmc,'-o', 'LineWidth', 2);  %3
plot(missRates,NMAE_kfmc,'-o', 'LineWidth', 2);  %4
plot(missRates,NMAE_pmc,'-o', 'LineWidth', 2);  %9
plot(missRates,NMAE_ablation,'-p', 'LineWidth', 2);  %1
plot(missRates,NMAE_our,'-p', 'LineWidth', 2);  %1
set(gca, 'LineWidth', 1, 'FontSize', 16); %坐标轴线条宽度 & 坐标轴字体大小
xlabel('Missing Rate','FontSize',20);
ylabel('NMAE','FontSize',20);
title('PM 2.5','FontSize',20);
legend('NMF','SVT','VMC','LADMC','KFMC','PMC','Ablation','Proposed')
ll = legend;
ll.FontSize = 14;
grid on;