
function [X,Y,tau,feat,avgError,assignment,assignmentinv] = matchlayer(X,Y,feat,tau,k,param)
    % matchlayer - substep of the main method for one shell pair
    %
    % X,Y: input shape
    % feat: dynamic exchange collection
    % a: current deformation coefficients
    % k: current shell size
    % param: parameter values

    % 调用 smoothshapesigmoid 函数对输入形状 X 进行平滑处理，得到平滑后的形状 XSmooth 和基础权重 feat.basisWeights
    [XSmooth,feat.basisWeights] = smoothshapesigmoid(X,k);
    % 调用 smoothshapesigmoid 函数对目标形状 Y 进行平滑处理，得到平滑后的形状 YSmooth
    [YSmooth,~] = smoothshapesigmoid(Y,k);

    avgError = 0;
    % 创建一个大小为 (X.n*3, X.n*3) 的单位矩阵 JleftCurr，其中 X.n 是顶点数量
    JleftCurr = speye(X.n*3);

    % 循环迭代参数 param.numSub 次，进行子步骤处理
    for iSub = 1:param.numSub

        %compute morphed shape from previous iteration
        % rotatevertices(JleftCurr, X.evecs(:,1:k) * tau(1:k,:)) 对 X.evecs(:,1:k) 乘以当前的变形系数 tau，并应用旋转操作
        % 结果是变形形状的顶点坐标 vertCurrFull
        vertCurrFull = XSmooth.vert + rotatevertices(JleftCurr,X.evecs(:,1:k) * tau(1:k,:));
        % 从变形形状中提取子样本 X.samples 的顶点坐标，得到 vertCurr
        vertCurr = vertCurrFull(X.samples,:);

        % 计算变形形状 vertCurrFull 的法向量
        feat.normalCurr = compute_normal(vertCurrFull,X.triv',X.flipNormal)';

        %compute correspondences
        % 将当前子步骤的索引存储在 feat.iSub 中
        feat.iSub = iSub;
        % 调用 computecorrespondences 函数计算形状之间的对应关系
        % 返回结果包括 assignment 和 assignmentinv，它们分别表示形状 XSmooth 到形状 YSmooth 和形状 YSmooth 到形状 XSmooth 的对应关系。
        [assignment,assignmentinv,feat] = computecorrespondences(XSmooth,YSmooth,vertCurr,vertCurrFull,param,feat);

        feat.assignment = assignment;
        feat.assignmentinv = assignmentinv;

        %compute the linear system for the data energy term
        
        % 计算用于数据能量项的线性系统
        % X.evecs(X.samples,1:k)' * X.evecs(X.samples,1:k) 是数据项的第一部分
        % X.evecs(assignmentinv,1:k)' * X.evecs(assignmentinv,1:k) 是数据项的第二部分
        % 结果存储在矩阵 M 中
        M = X.evecs(X.samples,1:k)' * X.evecs(X.samples,1:k) + ...
            X.evecs(assignmentinv,1:k)' * X.evecs(assignmentinv,1:k);
        % 计算线性系统的右侧项
        % -X.evecs(X.samples,1:k)' * (XSmooth.vertSub - YSmooth.vert(assignment,:)) 是数据项的第一部分
        % -X.evecs(assignmentinv,1:k)' * (XSmooth.vert(assignmentinv,:) - YSmooth.vertSub) 是数据项的第二部分
        % 结果存储在向量 z 中
        z = -X.evecs(X.samples,1:k)' * (XSmooth.vertSub - YSmooth.vert(assignment,:)) ...
            -X.evecs(assignmentinv,1:k)' * (XSmooth.vert(assignmentinv,:) - YSmooth.vertSub);



        %compute the Gauss Newton system for the arap regularization term
        if param.lambdaArap > 0

            % 调用 determinerotationsshifted 函数计算变形形状的旋转矩阵 feat.RCurr
            feat.RCurr = determinerotationsshifted(XSmooth,vertCurrFull);
            % 调用 transform_SO3_so3 函数将旋转矩阵 feat.RCurr 转换为李代数形式 feat.wCurr
            feat.wCurr = transform_SO3_so3(feat.RCurr);

            % 调用 jacobianshiftarap 函数计算 ARAP 正则化项的雅可比矩阵和右侧项
            % 返回结果为雅可比矩阵 Marap 和右侧项 zarap
            [Marap,zarap] = jacobianshiftarap(XSmooth.vert,feat.wCurr,X.evecs(:,1:k),X.neigh,(1:X.n)');

            % 计算当前的 ARAP 正则化项的权重 lambdaArapCurr
            % norm(M,inf) 计算矩阵 M 的无穷范数
            % norm(Marap,inf) 计算矩阵 Marap 的无穷范数
            lambdaArapCurr = param.lambdaArap * norm(M,inf) ./ norm(Marap,inf);

            % 将 ARAP 正则化项添加到线性系统的矩阵 M 中
            % lambdaArapCurr .* Marap 是加权的雅可比矩阵
            M = M + lambdaArapCurr .* Marap;
            % 将 ARAP 正则化项添加到线性系统的右侧项 z 中
            % lambdaArapCurr .* zarap 是加权的右侧项
            z = z + lambdaArapCurr .* zarap;

        end

        %solve the linear system
        % 解线性系统 M * tau = z，得到变形系数 tau
        % 更新变形系数 tau 的前 k 个分量
        tau(1:k,:) = M \ z;


        %compute the new functional map
        % 调用 computefunctionalmap 函数计算功能映射
        [feat,C,Cgt] = computefunctionalmap(X,Y,param,feat,k,assignment,assignmentinv);

        %plot intermediate results (if settings demand it)
        if ~param.noPlotInBetween
            % 计算平均误差，即子样本 X.samples 与对应顶点 Y.vert(assignment,:) 之间的欧氏距离的均值
            avgError = mean(normv(Y.vert(X.samples,:)-Y.vert(assignment,:)));
            plotskeletonlayered(X,Y,XSmooth,YSmooth,param,assignment,assignmentinv,tau,avgError,k,C,Cgt);

        end
    end

    %output error, if gt is available
    % 检查变形形状的顶点数量是否与目标形状 YSmooth 的顶点数量相同，且参数 param.intermediateOutput 为真
    if size(vertCurrFull,1)==size(YSmooth.vert,1) && param.intermediateOutput
        % 显示最大误差和平均误差
        % 计算变形形状 vertCurrFull 与目标形状 YSmooth.vert 之间的欧氏距离的最大值和均值
        disp('max error: ' + string(max(normv(vertCurrFull-YSmooth.vert))) + ', mean error: ' + string(mean(normv(vertCurrFull-YSmooth.vert))))
    end

    

    %plot intermediate results (if settings demand it)
    if ~param.noPlot

        plotskeletonlayered(X,Y,XSmooth,YSmooth,param,assignment,assignmentinv,tau,avgError,k,C,Cgt);

    end
end
