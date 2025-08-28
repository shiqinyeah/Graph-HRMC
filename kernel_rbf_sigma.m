function KXY=kernel_rbf_sigma(sigma,X,Y)
KXY=zeros(size(X,2),size(Y,2));
for i=1:size(X,2)
    for j=1:size(Y,2)
        temp=norm(X(:,i)-Y(:,j),2)^2;
        sigma2=2*sigma^2;
        KXY(i,j)=exp(-temp/sigma2);
    end
end
end