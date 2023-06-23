% 返回对齐后的形状X
function [X] = alignrigid(X,Y,param)
    % alignrigid - compute the rigid alignment of two shapes
    %
    % X,Y: input shapes
    % param: parameter values
    

    % 创建一个长度为6的数组kArrayTestSkeleton，其中包含数字1到6
    kArrayTestSkeleton = 1:6;
    
    %% main axis alignment

    % 将形状X和Y分别平滑化为XSmooth2和YSmooth2
    XSmooth2 = smoothshape(X,2);
    YSmooth2 = smoothshape(Y,2);
    
    

    % 将XSmooth2和YSmooth2的顶点坐标减去均值，并除以均值的平方
    XSmooth2.vert = (XSmooth2.vert-mean(XSmooth2.vert));
    YSmooth2.vert = (YSmooth2.vert-mean(YSmooth2.vert));
    XSmooth2.vert = XSmooth2.vert ./ mean(XSmooth2.vert.^2);
    YSmooth2.vert = YSmooth2.vert ./ mean(YSmooth2.vert.^2);
    

    % 从XSmooth2和YSmooth2中提取第二个主成分方向，并存储在dir1和dir2中
    dir1 = XSmooth2.xi(2,:)';
    dir2 = YSmooth2.xi(2,:)';

    % 将dir1和dir2向量归一化
    dir1 = dir1 ./ norm(dir1);
    dir2 = dir2 ./ norm(dir2);

    % 根据dir1和dir2之间的夹角，选择合适的旋转矩阵Rw
    if acos(dir1' * dir2) <= pi/2
        Rw = rotvectorpairs(dir1',dir2');
    else
        Rw = rotvectorpairs(dir1',-dir2');
    end

    % 计算XSmooth2和YSmooth2的顶点的平均值，分别存储在midpointCurrX和midpointCurrY中
    midpointCurrX = mean(XSmooth2.vert,1);
    midpointCurrY = mean(YSmooth2.vert,1);

    % 根据平移、中心点平移和旋转矩阵计算齐次变换矩阵xi
    xiTranslate = transform_SE3_se3_homCoords([eye(3),midpointCurrX';zeros(1,3),1]);
    xiMidpoints = transform_SE3_se3_homCoords([eye(3),midpointCurrY' - midpointCurrX';zeros(1,3),1]);
    
    xiRot = transform_SE3_se3_homCoords(Rw);
    xi = groupMult(groupMult(xiTranslate,xiMidpoints),groupMult(xiRot,-xiTranslate));

    % 将形状X的顶点根据xi进行刚性变换
    X.vert = rigidTransform(X.vert',xi')';
    

    % 通过形状的特征向量和特征值，对XSmooth和YSmooth进行进一步处理
    XSmooth.xi = (X.evecs(:,kArrayTestSkeleton)' * X.A * X.vert);
    XSmooth.vert = X.evecs(:,kArrayTestSkeleton) * XSmooth.xi;
    YSmooth.xi = (Y.evecs(:,kArrayTestSkeleton)' * Y.A * Y.vert);
    YSmooth.vert = Y.evecs(:,kArrayTestSkeleton) * YSmooth.xi;
    XSmooth.vert = (XSmooth.vert-mean(XSmooth.vert));
    YSmooth.vert = (YSmooth.vert-mean(YSmooth.vert));
    XSmooth.vert = XSmooth.vert ./ mean(XSmooth.vert.^2);
    YSmooth.vert = YSmooth.vert ./ mean(YSmooth.vert.^2);
    XSmooth.triv = X.triv;
    YSmooth.triv = Y.triv;
    
    
    %% ICP around main axis

    % 使用最远点采样(FPS)算法从XSmooth和YSmooth中选择一定数量的顶点样本，并进行排序
    problemSize = 1000;
    
    samplesX = fps_euclidean(XSmooth.vert, problemSize, randi(X.n));
    samplesX = sort(samplesX);

    samplesY = fps_euclidean(YSmooth.vert, problemSize, randi(Y.n));
    samplesY = sort(samplesY);


    % 根据ICP算法，使用X和Y的采样点进行迭代的刚性对齐，并更新X的顶点坐标
    xiICP = computeAxisICP(X.vert(samplesX,:),Y.vert(samplesY,:),5,YSmooth.xi(2,:));
    X.vert = rigidTransform(X.vert',xiICP')';
    
    % 根据权重调整X的顶点坐标，使其在平均值的加权下保持一致
    weightAX = diag(X.A) ./ mean(diag(X.A));
    weightAY = diag(Y.A) ./ mean(diag(Y.A));
    X.vert = X.vert - mean(X.vert .* weightAX,1) + mean(Y.vert .* weightAY,1);
    
    
    
    %% surrogate runs
    
    %params init
    % 使用sigmoid函数将X和Y的顶点平滑化，并计算其法线
    kMax = 500;
    
    XSmooth = smoothshapesigmoid(X,kMax);
    YSmooth = smoothshapesigmoid(Y,kMax);
    
    [X.normal,~,X.flipNormal] = compute_normal(XSmooth.vert',XSmooth.triv');
    [Y.normal,~,Y.flipNormal] = compute_normal(YSmooth.vert',YSmooth.triv');
    X.normal = X.normal';
    Y.normal = Y.normal';
    
    

    % 对一些变量进行初始化，如XsamplesOLD和YsamplesOLD保存原始的采样点索引，param和ainit是参数，problemSize是问题的大小，
    % X.samples和Y.samples存储新的采样点索引，X.vertSub和Y.vertSub存储新的采样点坐标，feat是一个结构体，kTest和kArray用于计算功能映射
    XsamplesOLD = X.samples;
    YsamplesOLD = Y.samples;
    
    param = standardparams(param);
    param.facFeat = 1000;
    param.noPlot = true;
    ainit = zeros(1,3);
    
    
    problemSize = param.problemSizeInit;
    X.samples = fps_euclidean(X.vert, problemSize, randi(X.n));
    X.samples = sort(X.samples);
    Y.samples = fps_euclidean(Y.vert, problemSize, randi(Y.n));
    Y.samples = sort(Y.samples);
    X.vertSub = X.vert(X.samples,:);
    Y.vertSub = Y.vert(Y.samples,:);
    
    
    feat = struct;
    feat.wCurr = zeros(X.n,3);
    
    kTest = 20;
    
    [~,feat.basisWeights] = smoothshapesigmoid(X,kTest);
    feat.oldBasisWeights = feat.basisWeights;
    
    kArray = 2:kTest;
    [feat] = computefunctionalmap(X,Y,param,feat,kTest);
    
    
    
    
    %first surrogate runs for vertical swap
    % 通过主成分分析(PCA)计算X的主方向，并调用surrogateRun函数进行垂直交换的代理运行
    [coeffX,~,~] = pca(X.vert);
    principalDirX = coeffX(:,3);
    
    X = surrogateRun(X,Y,feat,param,kArray,principalDirX,2);
    
    
    %init for second run
    % 更新kTest和kArray的值，并再次调用computefunctionalmap函数计算功能映射
    kTest = 20;
    
    [~,feat.basisWeights] = smoothshapesigmoid(X,kTest);
    feat.oldBasisWeights = feat.basisWeights;
    
    kArray = 2:kTest;
    [feat] = computefunctionalmap(X,Y,param,feat,kTest);
    
    
    %second surrogate runs for main axis rotation
    % 通过PCA计算X的主方向，并调用surrogateRun函数进行主轴旋转的代理运行
    [coeffX,~,~] = pca(X.vert);
    principalDirX = coeffX(:,1);
    
    X = surrogateRun(X,Y,feat,param,kArray,principalDirX,4);
    
    

    % 恢复原始的采样点索引，并对X和Y进行平滑化和法线计算
    X.samples = XsamplesOLD;
    Y.samples = YsamplesOLD;
    
    
    kMax = 500;
    XSmooth = smoothshapesigmoid(X,kMax);
    YSmooth = smoothshapesigmoid(Y,kMax);

    [X.normal,~,X.flipNormal] = compute_normal(XSmooth.vert',XSmooth.triv');
    [Y.normal,~,Y.flipNormal] = compute_normal(YSmooth.vert',YSmooth.triv');
    X.normal = X.normal';
    Y.normal = Y.normal';


    
    % 函数接受X、Y、feat、param、kArray、principalDirX和numSurr作为输入，并返回更新后的形状X。在函数中，
    % 通过循环调用rigidTransform函数对X进行主轴旋转，然后根据权重调整顶点坐标，平滑化形状，计算功能映射，并保存每次迭代的结果。
    % 最后返回最后一次迭代的结果
    function X = surrogateRun(X,Y,feat,param,kArray,principalDirX,numSurr)
    
        kMax = 500;
        
        param = standardparams(param);
        param.lambdaLap = 1;
        param.numSub = 5;
        param.mode = 1;
        param.facFeat = 0.11;
        param.normalDamping = 0.1;
        param.intermediateOutput = false;
        
        ainit = zeros(1,3);

        Xsaves = cell(numSurr,1);

        for iSurr = 1:numSurr

            X.vert = rigidTransform(X.vert',[0,0,0,principalDirX' .* 2 .* pi .* 1 ./ numSurr]')';
            
            weightAX = diag(X.A) ./ mean(diag(X.A));
            weightAY = diag(Y.A) ./ mean(diag(Y.A));
            X.vert = X.vert - mean(X.vert .* weightAX,1) + mean(Y.vert .* weightAY,1);

            Xsaves{iSurr} = X;


            XSmooth = smoothshapesigmoid(X,kMax);
            YSmooth = smoothshapesigmoid(Y,kMax);

            [X.normal,~,X.flipNormal] = compute_normal(XSmooth.vert',XSmooth.triv');
            [Y.normal,~,Y.flipNormal] = compute_normal(YSmooth.vert',YSmooth.triv');
            X.normal = X.normal';
            Y.normal = Y.normal';

            %% compute matching

            [a,~,~,featOut] = layeredmatching(X,Y,feat,kArray,ainit,param);


            vertCurrOrig = shiftedvertupsample(X,a);
            featOut.normalCurr = compute_normal(vertCurrOrig,X.triv',X.flipNormal)';


            %% compute error

            paramEval = param;
            paramEval.matchingAlignment = true;


            featOut = computeCorrCurr(X,Y,vertCurrOrig,paramEval,featOut);


            
            errorArray(iSurr) = mean([diag(X.A(X.samples,X.samples)) .* featOut.Dass;diag(Y.A(Y.samples,Y.samples)) .* featOut.Dassinv]);
            

            if size(vertCurrOrig,1) == size(Y.vert,1)
                currErrorGT = max(normv(vertCurrOrig-Y.vert));
                errorArrayGT(iSurr) = currErrorGT;
            end
        end

        iMin = find(errorArray == min(errorArray),1);

        X = Xsaves{iMin};
    end

    function featOut = computeCorrCurr(XSmooth,YSmooth,vertCurrFull,paramEval,featOut)
        [assignment,~,featOut] = computecorrespondences(XSmooth,YSmooth,vertCurrFull(XSmooth.samples,:),vertCurrFull,paramEval,featOut);
        featOut.assignment = assignment;
    end
end











