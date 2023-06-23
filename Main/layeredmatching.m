function [tau,X,Y,feat] = layeredmatching(X,Y,feat,kArray,aInit,param)
    % layeredmatching - main registration method for two input shapes X,Y
    %
    % X,Y: input shape
    % feat: dynamic exchange collection
    % kArray: coarse-to-fine shell sizes 
    % aInit: deformation initialization from initialization
    % param: parameter values

    % 检查是否存在参数param，如果不存在则使用默认的参数值
    if ~exist('param','var')
        param = standardparams();
    end

    % 将参数param.noPlotInBetween设置为true，表示不在中间步骤中绘制结果
    param.noPlotInBetween = true;
    % 将参数param.plotCorr设置为false，表示不绘制对应关系图。
    param.plotCorr = false;
    % 将参数param.twoPlots设置为false，表示不绘制两个形状的比较图
    param.twoPlots = false;


    % 创建大小为(kArray的最后一个元素, 3)的零矩阵tau，用于存储形状的变形系数
    tau = zeros(kArray(end),3);
    % 检查是否存在变量aInit，即是否存在从初始化中得到的变形初始化值
    if exist('aInit','var')
        % 将变形初始化值aInit复制到tau的前size(aInit,1)行
        tau(1:size(aInit,1),:) = aInit;
    end

    % 创建一个大小为(X.n, 3)的零矩阵feat.wCurr，用于存储当前变形的旋转参数
    feat.wCurr = zeros(X.n,3);

    % 将参数param.normalDamping的值赋给变量normalDamping，表示法向量的阻尼系数
    normalDamping = param.normalDamping;

    %% coarse-to-fine matching - all steps
    % 对于每个kArray中的元素，执行以下操作
    for iK = 1:length(kArray)

        % 如果参数param.intermediateOutput为真，则显示当前的k值
        if param.intermediateOutput
            disp('k = ' + string(kArray(iK)));
        end

        % 获取当前迭代的k值
        k = kArray(iK);

        % 计算当前的ARAP正则化项权重param.lambdaArap
        % param.lambdaArapInit为初始权重
        % sqrt(max(k-20,0))用于根据k值调整权重
        param.lambdaArap = param.lambdaArapInit .* sqrt(max(k-20,0));
        % 根据k值设置法向量的阻尼系数
        % 如果k大于10，则使用原始阻尼系数normalDamping，否则设置为0
        param.normalDamping = normalDamping .* (k > 10);

        % if a functional map from the initialization is used, it should
        % only be used until the 20th step for maximum accuracy
        % 检查当前k是否大于20且feat结构中存在字段Cfix
        if k > 20 && isfield(feat,'Cfix')
            % 如果满足条件，从feat结构中移除字段Cfix
            feat = rmfield(feat,'Cfix');
        end
        
        %call sub matching step
        % 调用matchlayer函数进行子匹配步骤，计算形状之间的对应关系，解线性系统以获得形状的变形系数
        [X,Y,tau,feat] = matchlayer(X,Y,feat,tau,k,param);
        
    end







    
    
    
    
        
        
        
        
