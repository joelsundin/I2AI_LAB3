function gc_task()
    % An example of how to segment a color image according to pixel colors.
    % Fisrt stage identifies k distinct clusters in the color space of the
    % image. Then the image is segmented according to these regions; each pixel
    % is assigned to its cluster and the GraphCut poses smoothness constraint
    % on this labeling.

    close all

    addpath('GCmex/');
    %check for mex files, compile if not found
    if(~(exist('GraphCut3dConstr.mexw64') && ...
        exist('GraphCutConstr.mexw64') && ...
        exist('GraphCutConstr.mexw64') && ...
        exist('GraphCutConstr.mexw64')))
        oldFolder = cd('./GCmex');
        compile_gc()
        cd(oldFolder);
    end

    % read an image
    im = im2double(imread('parrot.jpg'));
    sz = size(im);
    imtool(im)
    % cluster the image colors into k regions
    data = ToVector(im);
    %[idx1, c] = kmeans(data, k);
    idx   = zeros(sz(1),sz(2),'uint8');
    cen = [];
    srcCords = [];
    snkCords = [];
    k = 0;
    
    % background
    %apparently y1 x1, y2 x2
    cords = [10, 1389, 679, 1888];


    k = k+1;
    [idx,c] = addSrcSnk(im,idx,cords,k, 'background');
    cen = [cen; c];
    snkCords = [snkCords;cords];
    

    % foreground
    cords = [355, 1009, 754, 1198];
    k = k+1;
    [idx,c] = addSrcSnk(im,idx,cords,k, 'parrot');
    cen = [cen; c];
    srcCords = [srcCords;cords];
    
    % calculate the data cost per cluster center
    Dc = zeros([sz(1:2) k],'single');
    for ci=1:k
        % use covariance matrix per cluster
        icv = inv(cov(data(idx==ci,:)));    
        dif = data - repmat(cen(ci,:), [size(data,1) 1]);
        % data cost is minus log likelihood of the pixel to belong to each
        % cluster according to its RGB value
        Dc(:,:,ci) = reshape(sum((dif*icv).*dif./2,2),sz(1:2));
    end

    % cut the graph

    % smoothness term: 
    % constant part
    Sc = ones(k) - eye(k);
    % spatialy varying part
    [Hc,Vc] = SpatialCues(im);

    scsnk_wgt = 10.0;
    edges_wgt = 5.0;
    dc_wgt = 1.0;
    gch = GraphCut('open', dc_wgt.*Dc, scsnk_wgt.*Sc, exp(-Vc*edges_wgt), exp(-Hc*edges_wgt));
    [gch,L] = GraphCut('expand',gch);
    gch = GraphCut('close', gch);

    % show results
    segImg = im;
    
    %Seperate Src regions and Snk regions
    for i = 1:size(snkCords,1)
        cords    = snkCords(i,:);
        patch    = L(cords(1):cords(3),cords(2):cords(4));
        clsLabel = median(reshape(patch,[1,numel(patch)]));
        segImg(:,:,1) = segImg(:,:,1).*(~(L == clsLabel));
        segImg(:,:,2) = segImg(:,:,2).*(~(L == clsLabel));
        segImg(:,:,3) = segImg(:,:,3).*(~(L == clsLabel));
        Lmask = ~(L == clsLabel);
    end
    
    % Only supports ground-truth evaluation with 2 classes
    if k == 2
        % Load ground-truth
        gt = imread('parrot_ground_truth_gray.png');
        % Compute masks
        gt_mask = (gt > 0);
        
        % Compute ground-truth parrot to background ratio
        fprintf('Parrot to background pixel ratio: %f.\n', mean(gt_mask(:)));
                
        % Compute dice coefficients of result
        dice_coeff = dice(Lmask, gt_mask);
        
        errImg = (Lmask ~= gt_mask);
        
        figure('Name', 'Segmentation Error Image');
        imshow(errImg);
        
        % Print dice coefficient
        fprintf('Dice: %f\n', dice_coeff);    

        % Error image
        imwrite(errImg,'errOut.png');    
        imwrite(uint8(~Lmask)*255,'background.png');    
        imwrite(uint8(Lmask)*255,'foreground.png');    
    end
    
    figure('Name', 'Masked Parrot');
    imshow(segImg);

    imwrite(segImg,'bOut.png');
    disp('Done');
end

%---------------- Aux Functions ----------------%
function v = ToVector(im)
    % takes MxNx3 picture and returns (MN)x3 vector
    sz = size(im);
    v = reshape(im, [prod(sz(1:2)) 3]);
    vv = v;
    %row major vs col major access variation
    v(:,1) = vv(:,3);
    v(:,3) = vv(:,1);
end

function [idx,cen] = addSrcSnk(img,idx,bBox,nCen,objectName)
    imgPatch = img(bBox(1):bBox(3),bBox(2):bBox(4),:);
    figure('Name', objectName);
    imshow(imgPatch);
    %title(objectName);
    dt = ToVector(imgPatch);
    [is,cen] = kmeans(dt, 1);
    idx(bBox(1):bBox(3),bBox(2):bBox(4)) = nCen;
end

%-----------------------------------------------%
function [hC,vC] = SpatialCues(im)
    g = fspecial('gauss', [13 13], sqrt(13));
    dy = fspecial('sobel');
    vf = conv2(g, dy, 'valid');
    sz = size(im);

    vC = zeros(sz(1:2));
    hC = vC;

    for b=1:size(im,3)
        vC = max(vC, abs(imfilter(im(:,:,b), vf, 'symmetric')));
        hC = max(hC, abs(imfilter(im(:,:,b), vf', 'symmetric')));
    end
end
    