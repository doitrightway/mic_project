% A = load('data/assignmentSegmentBrain.mat');
% orig_img = A.imageData;

% rand('seed', 1);


orig_imgg = zeros(128,128,5);
test_label = zeros(128,128,5);

lim=5;
num_sgmnt = 5;


test = phantom(128);
orig_imgg(:,:,1) = phantom(128);
ground = reshape(kmeans(test(:),5), 128,128);
for i=1:5
    test_label(:,:,i) = ground==i;
end

meanarr=zeros(num_sgmnt,1);
for j=1:num_sgmnt
    meanarr(j)=sum(sum(test_label(:,:,j).*phantom(128)))/(128*128);
end
testlabels= zeros(128,128,num_sgmnt);
B=sort(meanarr);
for k=1:num_sgmnt
    c=find(meanarr==B(k));
    testlabels(:,:,k)=test_label(:,:,c);
end
orig_imgg(:,:,2) = imnoise(phantom(128), 'gaussian', 0, 0.0001);
orig_imgg(:,:,3) = imnoise(phantom(128), 'gaussian', 0, 0.0005);
orig_imgg(:,:,4) = imnoise(phantom(128), 'gaussian', 0, 0.0025);
orig_imgg(:,:,5) = imnoise(phantom(128), 'gaussian', 0, 0.01);

% orig_img1 = imread('data/elephant.jpg');
    % orig_img2=rgb2gray(orig_img1);
    % orig_img=double(orig_img2)./255;

    % orig_img = [0.25 0.45 0.75 0.9; 0.45 0.34 0.56 0.32; 0.23 0.95 0.13 0.56; 0.11 0.06 0.65 0.43];

for ii=1:lim
    orig_img = orig_imgg(:,:,ii);
    imshow(orig_img);
    % figure;
    size1 = size(orig_img, 1);
    size2 = size(orig_img, 2);

    num_sgmnt = 5;

    % list_algo = ['k_means', 'fuzzy_C_means', 'EM', 'minCut'];

    num_algo = 4;

    labels = zeros(size1, size2, num_sgmnt, num_algo);

    display('Iamhere1');
    labels(:,:,:,1) = k_means(orig_img, num_sgmnt);
    % display('Iamhere2');
    labels(:,:,:,2) = fuzzy_C_means(orig_img, num_sgmnt);
    % display('Iamhere3');
    labels(:,:,:,3) = EM(orig_img, num_sgmnt);
    % display('Iamhere4');
    labels(:,:,:,4) = minCut(orig_img, num_sgmnt);
%     display('Iamhere5');
%     labels(:,:,:,1) = genetic(orig_img, num_sgmnt);

    for i=1:num_algo
        figure;
        for j=1:num_sgmnt
            subplot(1,num_sgmnt,j);
            imshow(labels(:,:,j,i));
        end
    end

    diff = zeros(num_algo, 1);
    for i=1:num_algo
        meanarr=zeros(num_sgmnt,1);
        for j=1:num_sgmnt
            meanarr(j)=sum(sum(labels(:,:,j,i).*orig_img))/(size1*size2);
        end
          newlabels= zeros(size1,size2,num_sgmnt);
        [B,I]=sort(meanarr);
        for k = 1:num_sgmnt
            newlabels(:,:,k)=labels(:,:,I(k),i);
        end
%         for k=1:num_sgmnt
%             c=find(meanarr==B(k));
%             for p=1:length(c)
%                 newlabels(:,:,k,i)=labels(:,:,c(p),i);
%             end
%         end
        diff(i) = sum(sum(sum(abs(newlabels-testlabels))));
    end
    figure;
    plot(1:num_algo,diff);
    xlabel('Different Algorithms');
    ylabel('Difference values');
    figure;
end