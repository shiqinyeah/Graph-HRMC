%% 3. PM2.5  
load('D:\Dataset\TianJiazheng\PM2.5\PM25_tensor.mat')
T=double(Y_source);
[s1, s2, s3] = size(T);
GroundTruth=[];
for k= 1:70
    slice = squeeze(T(:,:,k));
    GroundTruth=[GroundTruth slice];
end

%% Normalization
min_value = min(GroundTruth(:));  
max_value = max(GroundTruth(:));  
GROUNDTRUTH = (GroundTruth - min_value) / (max_value - min_value);
[m,n]=size(GROUNDTRUTH);

%% Feature mapping
X=GROUNDTRUTH;
[M,N] = size(X);
phiX = zeros(nchoosek(M+1,2),N);  % 2阶齐次
trimask = logical(triu(ones(M),0)); % upper triagular mask
for j = 1:N
    x = X(:,j);
    x2 = x*x';
    phiX(:,j) = x2(trimask);
end
Y=[ones(1,N);X;phiX];

threshold=99;

%% power curve 1
[~, S1, ~] = svd(GROUNDTRUTH);
singular_values1 = diag(S1);
energy1 = cumsum(singular_values1.^2) / sum(singular_values1.^2)*100;
thresholdIndex1 = find(energy1 >= threshold, 1);
rdr1=thresholdIndex1/length(singular_values1)*100;

%% power curve 2
[~, S2, ~] = svd(Y);
singular_values2 = diag(S2);
energy2 = cumsum(singular_values2.^2) / sum(singular_values2.^2)*100;
thresholdIndex2 = find(energy2 >= threshold, 1);
rdr2=thresholdIndex2/length(singular_values2)*100;

xmin=min(rdr1,rdr2);
xmax=max(rdr1,rdr2);
ymin=min(energy1(thresholdIndex1),energy2(thresholdIndex2));
ymax=max(energy1(thresholdIndex1),energy2(thresholdIndex2));

x1=(1:length(singular_values1))./length(singular_values1)*100;
x2=(1:length(singular_values2))./length(singular_values2)*100;

figure(1);
plot(x1,energy1,'-g*','LineWidth', 1); hold on;
plot(x2,energy2,'-b*','LineWidth', 1); hold on;
plot(rdr1, energy1(thresholdIndex1), 'ro', 'MarkerSize', 8, 'LineWidth', 2); hold on;
plot(rdr2, energy2(thresholdIndex2), 'ro', 'MarkerSize', 8, 'LineWidth', 2); hold on;
set(gca, 'LineWidth', 1, 'FontSize', 16); %坐标轴线条宽度 & 坐标轴字体大小
title('PM 2.5','FontSize', 20);
xlabel('Rank Dimension Ratio','Interpreter','tex','FontSize', 20);
ylabel('Cumulative Energy','FontSize', 20);
xLimits = xlim; % 获取 x 轴区间
yLimits = ylim; % 获取 y 轴区间
text(xLimits(2)+1, yLimits(1), '%', 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', 'FontSize', 16);
text(xLimits(1), yLimits(2), '%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 16);
grid on;
legend('q=1','q=2')
l = legend;
l.FontSize = 14;
grid on;

axes('Position', [.4 .3 .4 .4]); % 创建一个新的坐标轴（子图）
box on;
% line 1
plot(x1,energy1,'-g*','LineWidth', 1); hold on;
plot(rdr1, energy1(thresholdIndex1), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
set(gca, 'LineWidth', 1, 'FontSize', 16); %坐标轴线条宽度 & 坐标轴字体大小
xlim([max(xmin-5,0), xmax+0.5]);
ylim([ymin-1, min(100,ymax+1)]);
text(xmax+0.55,ymin-1, '%', 'VerticalAlignment', 'middle', 'HorizontalAlignment', 'left', 'FontSize', 16);
text(max(xmin-5,0),min(100,ymax+1), '%', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center', 'FontSize', 16);
grid on;
set(gca, 'XLabel', [], 'YLabel', [], 'Title', []);
line([rdr1 rdr1], [ymin-1 energy1(thresholdIndex1)], 'color','k', 'LineStyle', '--','LineWidth', 0.8);
line([max(xmin-5,0) rdr1], [energy1(thresholdIndex1) energy1(thresholdIndex1)], 'color','k', 'LineStyle', '--','LineWidth', 0.8);
text(rdr1, energy1(thresholdIndex1)-0.2, sprintf('(%.2f%%, %.2f%%)', rdr1, energy1(thresholdIndex1)), 'VerticalAlignment', 'top', 'HorizontalAlignment', 'center','FontSize', 16);
% line 2
plot(x2,energy2,'-b*','LineWidth', 1); hold on;
plot(rdr2, energy2(thresholdIndex2), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
line([rdr2 rdr2], [ymin-1 energy2(thresholdIndex2)], 'color','k', 'LineStyle', '--','LineWidth', 0.8);
line([max(xmin-5,0) rdr2], [energy2(thresholdIndex2) energy2(thresholdIndex2)],'color','k', 'LineStyle', '--','LineWidth', 0.8);
text(rdr2, energy2(thresholdIndex2), sprintf('(%.2f%%, %.2f%%)', rdr2, energy2(thresholdIndex2)), 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left','FontSize', 16);
%