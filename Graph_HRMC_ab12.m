% 邻居数由阈值确定
function X=Graph_HRMC_ab12(GroundTruth,Boolean,parameter)
%% parameter
RANK=parameter.RANK;
sigma=parameter.sigma;
threshold=parameter.threshold; 
alpha2=parameter.alpha2;
beta2=parameter.beta2;
IterMax=parameter.IterMax;

%% initialization
[M,N]=size(GroundTruth);
Xmean=zeros(M,N);
U=zeros(M,RANK);
A=zeros(M,M); % 对称！
B=zeros(N,N); % 对称！

LogicalOmega=logical(Boolean);
epsilon = 1e-6;

%% initialization of X via column-row mean
Xr = mean(GroundTruth .* Boolean, 2, 'omitnan');
Xc = mean(GroundTruth .* Boolean, 1, 'omitnan');
for i=1:M
    for j=1:N
        Xmean(i,j)=(Xr(i)+Xc(j))/2;
    end
end
X1=GroundTruth.*Boolean;  % 观测值
X2=Xmean.*(1-Boolean);    % 用均值填充缺失值
X=X1+X2;

%% Graph_HRMC
for iter=1:IterMax
    DA=diag(sum(A,2));
    LA=DA-A;
    
    DB=diag(sum(B,1));
    LB=DB-B;
    
    %% Update V
    KUU=kernel_rbf_sigma(sigma,U,U);
    KXU=kernel_rbf_sigma(sigma,X,U);
    KUU_reg = KUU + epsilon * eye(size(KUU));
    V = sylvester(KUU_reg, beta2*LB, KXU');
    
    %% Update U
    Q1=V'.* KXU;
    Gama1=diag(ones(1,N)*Q1);
    Q2=V*V'.*KUU;
    Gama2=diag(ones(1,RANK)*Q2);
    E=alpha2*sigma^2*LA;
    F=Gama1+Q2-Gama2;
    E_reg = E + epsilon * eye(size(E));
    F_reg = F + epsilon * eye(size(F));
    U=sylvester(E_reg,F_reg,X*Q1);
    
    %% Update A: M*M
    DA = pdist2(U, U, 'squaredeuclidean'); % U的每一行代表一个数据点
%     sorted_DA = sort(DA, 2);
    for i = 1:M   % 对于A的每一行
        row = DA(i, :);
        [sortedrow, ~] = sort(row);
        k=find(sortedrow<threshold,1,'last'); % 邻居数由阈值确定
        if ~isempty(k)
            SAk=sum(sortedrow(1:k));
            alpha1=alpha2*(k/2*sortedrow(k+1)-0.5*SAk);
        else
            SAk=0;
            disp(['There are no neighbors in row' num2str(i)])
        end
        for j=1:M
            mid=SAk/k-row(j);
            total=alpha2/(2*alpha1)*mid+1/k;
            A(i,j)=max(total,0);
        end   
    end
    A=(A+A')/2;  
    clear k mid total 

    %% Update B: N*N
    DB = pdist2(V', V', 'squaredeuclidean'); % V的每一列代表一个数据点
%     sorted_DB = sort(DB, 1);
    for j = 1:N  % 对于B的每一列
        col = DB(:,j);
        [sortedcol, ~] = sort(col);
        k=find(sortedcol<threshold,1,'last');
        if ~isempty(k)
            SBk=sum(sortedcol(1:k));
            beta1=beta2*(k/2*sortedcol(k+1)-0.5*SBk);
        else
            SBk=0;
            disp(['There are no neighbors in column' num2str(j)])
        end
        for i=1:N
            mid=SBk/k-col(i);
            total=beta2/(2*beta1)*mid+1/k;
            B(i,j)=max(total,0);
        end
    end
    B=(B+B')/2;
    clear k mid total
    
    %% Update X
    Xold=X;
    X=U*Q1'*pinv(diag(ones(1,RANK)*Q1'));
    X(LogicalOmega)=GroundTruth(LogicalOmega);
    
    %% Stopping Criterion
        error=norm(X-Xold,'fro')/norm(Xold,'fro');
        if error<1e-3
            break
        end
end
% figure;
% plot(error);
end