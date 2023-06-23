function [Cfix,ainit,iMin,errorArrayGT,errorArray] = initMCMC(X,Y,param,kMin,kMax)

    % initMCMC - Markov chain Monte Carlo initialization strategy
    %
    % X,Y: input shapes
    % param: parameter values
    % kMin,kMax: range of shell sizes for surrogate runs

    % 检查是否存在参数kMin，如果不存在则将其设置为默认值6
    if ~exist('kMin','var')
        kMin = 6;
    end

    % 检查是否存在参数kMax，如果不存在则将其设置为默认值20
    if ~exist('kMax','var')
        kMax = 20;
    end

    % 创建一个包含从kMin到kMax的整数序列的数组kArray，用于表示要进行MCMC初始化的不同层级的shell大小
    kArray = kMin:kMax;

    % 调用standardparams函数，使用给定的参数param或默认参数，将其赋值给变量param
    param = standardparams(param);
    % 将参数param.facFeat设置为1.5，用于控制特征能量项的权重
    param.facFeat = 1.5;
    % 将参数param.lambdaFeat设置为0，表示特征能量项的正则化权重为0
    param.lambdaFeat = 0;
    % 将参数param.plotCorr设置为false，表示不绘制对应关系图
    param.plotCorr = false;
    % 将参数param.twoPlots设置为false，表示不绘制两个形状的比较图
    param.twoPlots = false;
    % 将参数param.mode设置为2，用于控制计算特征能量项时使用的模式
    param.mode = 2;
    % 将参数param.intermediateOutput设置为false，表示不在中间输出时显示结果
    param.intermediateOutput = false;

    
    % 将参数param.numMCMC的值赋给变量numProp，表示MCMC中的迭代次数
    numProp = param.numMCMC;
    
    feat = struct;
    % 创建一个大小为(X.n, 3)的零矩阵feat.wCurr，用于存储当前变形的旋转参数
    feat.wCurr = zeros(X.n,3);
    

    % 调用smoothshapesigmoid函数对形状X进行平滑处理，得到平滑后的形状XSmooth，并将平滑后的权重存储在feat.basisWeights中
    [XSmooth,feat.basisWeights] = smoothshapesigmoid(X,kMax);
    % 将feat.basisWeights的值赋给feat.oldBasisWeights，用于存储旧的权重
    feat.oldBasisWeights = feat.basisWeights;
    YSmooth = smoothshapesigmoid(Y,kMax);
    

    % 创建一个大小为(numProp, 1)的零向量errorArray，用于存储每次迭代中的误差
    errorArray = zeros(numProp,1);
    % 创建一个大小为(numProp, 1)的零向量errorArrayGT，用于存储每次迭代中的与真实形状的误差
    errorArrayGT = zeros(numProp,1);
    % 创建一个大小为(numProp, 1)的空单元格数组CfixCollection，用于存储每次迭代中的功能映射矩阵
    CfixCollection = cell(numProp,1);
    % 创建一个大小为(numProp, 1)的空单元格数组aCollection，用于存储每次迭代中的形状变形系数
    aCollection = cell(numProp,1);

    % 将参数param.problemSizeInit的值赋给变量problemSize，表示初始化时使用的问题规模
    problemSize = param.problemSizeInit;
    % 调用fps_euclidean函数在形状X的顶点中进行均匀采样，得到问题规模为problemSize的采样点，存储在X.samples中
    X.samples = fps_euclidean(X.vert, problemSize, randi(X.n));
    % 对X.samples中的采样点进行排序
    X.samples = sort(X.samples);
    % 调用fps_euclidean函数在形状Y的顶点中进行均匀采样，得到问题规模为problemSize的采样点，存储在Y.samples中
    Y.samples = fps_euclidean(Y.vert, problemSize, randi(Y.n));
    % 对Y.samples中的采样点进行排序
    Y.samples = sort(Y.samples);
    % 将形状X中采样点的顶点坐标存储在X.vertSub中
    X.vertSub = X.vert(X.samples,:);
    % 将形状Y中采样点的顶点坐标存储在Y.vertSub中
    Y.vertSub = Y.vert(Y.samples,:);
    
    if kMax > 20
        % 调用getNeighbors函数计算形状X中顶点之间的邻接关系矩阵，将结果存储在X.neigh.mat中
        X.neigh.mat = getNeighbors(X.vert');
        % 根据采样点的邻接关系矩阵，找到对应的行索引和列索引，存储在X.neigh.row和X.neigh.col中
        [X.neigh.row,X.neigh.col] = find(X.neigh.mat(X.samples,:));
    end
    
    %% MCMC
    % 迭代numProp次，进行MCMC初始化
    for iSet = 1:numProp
        %% create proposal shell
        disp("Prop #" + string(iSet) + "...")
        

        % 随机生成一个1或2的整数，表示当前的模式
        modeCurr = randi(2);
        % 创建一个包含0和10的数组lambdaLapSet，用于表示拉普拉斯正则化项的权重
        lambdaLapSet = [0,10];
        % 将根据当前模式选择的拉普拉斯正则化项的权重赋值给参数param.lambdaLap
        param.lambdaLap = lambdaLapSet(modeCurr);
        % 调用proposalshell函数生成一个初始的形状变形系数ainit，用于MCMC的提议
        ainit = proposalshell(X,kMin);

        % 将参数param的值赋给变量paramComp，用于在子匹配步骤中传递参数
        paramComp = param;
        % 将参数paramComp.noPlot设置为true，表示不进行绘制
        paramComp.noPlot = true;
        % 调用layeredmatching函数执行层次匹配，并返回形状变形系数a、中间结果和特征信息featOut
        [a,~,~,featOut] = layeredmatching(X,Y,feat,kArray,ainit,paramComp);

        % 调用shiftedvertupsample函数根据形状变形系数a计算原始形状的顶点坐标
        vertCurrOrig = shiftedvertupsample(X,a);
        % 调用compute_normal函数计算顶点的法向量，并将结果存储在featOut.normalCurr中
        featOut.normalCurr = compute_normal(vertCurrOrig,X.triv',X.flipNormal)';
        
        %% determine whether to accept the current proposal
        
        paramEval = param;
        % 将参数paramEval.matchingAlignment设置为true，表示进行匹配对齐
        paramEval.matchingAlignment = true;
        % 将参数paramEval.facFeat设置为0.11，用于控制特征能量项的权重
        paramEval.facFeat = 0.11;

        % 调用computeCorrCurr函数计算当前形状与目标形状的对应关系，并更新特征信息
        featOut = computeCorrCurr(X,Y,vertCurrOrig,paramEval,featOut);

        % 计算当前提议的误差，并将其存储在errorArray中
        errorArray(iSet) = 1e7 .* mean([diag(X.A(X.samples,X.samples)) .* featOut.Dass;diag(Y.A(Y.samples,Y.samples)) .* featOut.Dassinv]);

        % 检查当前提议的形状顶点数是否与目标形状的平滑后顶点数相同。如果是，则计算当前提议与目标形状的顶点坐标之间的平均距离，并将其存储在errorArrayGT中
        if size(vertCurrOrig,1) == size(YSmooth.vert,1)
            currErrorGT = mean(normv(vertCurrOrig-YSmooth.vert));
            errorArrayGT(iSet) = currErrorGT;
        end

        % 将特征信息中的C存储在CfixCollection中
        CfixCollection{iSet} = featOut.C;
        % 形状变形系数a存储在aCollection中
        aCollection{iSet} = a;

        % 查找errorArray中前iSet个元素中的最小值的索引，并将其赋给iMin
        iMin = find(errorArray(1:iSet) == min(errorArray(1:iSet)),1);
        
        if ~param.noPlot
            
            if iMin == iSet
                plotskeletonlayered(X,Y,XSmooth,YSmooth,param,0,0,a,0,kMax);
            end
            
            title('MCMC - init: objective = ' + string(errorArray(iSet)) + ', min objective = ' + string(errorArray(iMin)) + ', iMin = ' + string(iMin) + ', iCurr = ' + string(iSet) + "/" + string(numProp));
            drawnow
        end
        
        disp("...objective value: " + string(errorArray(iSet)))
    end
    
    %% choose best configuration
    % 查找errorArray中的最小值的索引，并将其赋给iMin
    iMin = find(errorArray == min(errorArray),1);

    % 从CfixCollection中获取对应于最小误差的Cfix
    Cfix = CfixCollection{iMin};
    % 从aCollection中获取对应于最小误差的ainit
    ainit = aCollection{iMin};
    

    
    
    function featOut = computeCorrCurr(X,Y,vertCurrOrig,paramEval,featOut)
        [assignment,~,featOut] = computecorrespondences(X,Y,vertCurrOrig(X.samples,:),vertCurrOrig,paramEval,featOut);
        featOut.assignment = assignment;
    end
end






