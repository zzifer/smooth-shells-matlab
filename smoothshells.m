function [POut,tauOut,COut,X,Y] = smoothshells(input1,input2,param)
% smoothshells - compute correspondences for a pair of input shapes
% input1,input2: either a mesh (as a struct) OR a file containing a mesh
% param (optional): hyperparameters of the method

% 如果param不存在，则创建一个空结构体param，并使用standardparams函数为其赋值
if ~exist('param','var')
    param = struct;
    param = standardparams(param);
end

% linspace(1, (param.kMax-20).^(1/4), param.kArrayLength)表示在1到(param.kMax-20).^(1/4)之间生成param.kArrayLength个等间距的值
% round(linspace(1,(param.kMax-20).^(1/4),param.kArrayLength))表示将生成的值四舍五入为整数
% 20+round(linspace(1, (param.kMax-20).^(1/4), param.kArrayLength))^4表示将上述四舍五入后的值分别进行四次方运算，并将结果加上20
% 将20和上述计算得到的数组连接在一起，形成kArray数组
kArray = [20,20+round(linspace(1,(param.kMax-20).^(1/4),param.kArrayLength).^4)];
feat = struct;

%% load Shape

disp('Load shapes and compute features..')

% 调用loadshapepair函数，将input1和input2作为参数传入，并将返回值分别赋给X和Y
[X,Y] = loadshapepair(input1,input2);

% 如果参数param的字段noPlot的值为false，则调用surf_pair函数，将X和Y作为参数传入，用于绘制形状对
if ~param.noPlot
    surf_pair(X,Y);
    drawnow
end

%% rigid alignment

% 如果参数noRigid的值为false，则打印提示信息，表示正在计算刚性对齐
if ~param.noRigid
    disp('Computing rigid alignment...')
    % 调用alignrigid函数，将X、Y和param作为参数传入，并将返回值赋给 X。该函数用于进行刚性对齐
    [X] = alignrigid(X,Y,param);
    % 如果noPlot的值为false，则调用surf_pair函数，将X和Y作为参数传入，用于绘制形状对
    if ~param.noPlot
        surf_pair(X,Y);
        drawnow
    end
end

%% MCMC

disp('MCMC initialization...')

% 将X、Y和param作为参数传入，并将返回值分别赋给 feat.Cfix 和 tauInit
[feat.Cfix,tauInit] = initMCMC(X,Y,param);

%% full matching

disp('Full run...')

% 调用layeredmatching函数，将X、Y、feat、kArray、tauInit和 param 作为参数传入，
% 并将返回值分别赋给 tau、X、Y 和 featOut。该函数用于进行分层匹配
[tau,X,Y,featOut] = layeredmatching(X,Y,feat,kArray,tauInit,param);

% 计算变形后的完整形状vertCurrFull，通过将 X.vert和X.evecs(:,1:size(tau,1))相乘，并加上tau
% X.evecs(:,1:size(tau,1))表示X的特征向量矩阵的前size(tau,1)列，其中 tau 是当前的变形系数
% tau 是当前的变形系数，是一个大小为 (k,3) 的矩阵，其中 k 是当前的尺度大小
% X.evecs(:,1:size(tau,1))*tau表示将特征向量矩阵的前size(tau,1)列与变形系数相乘，得到一个大小为(X.n,3)的矩阵，其中X.n是顶点的数量
% X.vert + X.evecs(:,1:size(tau,1))*tau表示将原始顶点坐标与变形后的坐标相加，得到变形后的顶点坐标vertCurrFull
vertCurrFull = X.vert + X.evecs(:,1:size(tau,1)) * tau;

 %% plot result

% 如果noPlot的值为false，则进行结果的可视化。根据是否存在参考形状Y.vert，选择不同的权重来绘制形状。最后绘制三个子图，分别显示变形后的形状、源形状和参考形状
if ~param.noPlot

    %gt exists
    if size(vertCurrFull,1)==size(Y.vert,1)
        weightsPlotX = normv(vertCurrFull-Y.vert);
        weightsPlotY = weightsPlotX;
    else
        weightsPlotX = X.vert(:,2);
        weightsPlotY = Y.vert(:,2);
    end

    subplot(1,3,1)
    hold off
    trisurf(X.triv,vertCurrFull(:,1),vertCurrFull(:,2),vertCurrFull(:,3),weightsPlotX);
    axis equal
    colorbar
    title('Morphed shape $\hat{\mathcal{X}}$','interpreter','latex')

    subplot(1,3,2)
    hold off
    trisurf(X.triv,X.vert(:,1),X.vert(:,2),X.vert(:,3),weightsPlotX);
    axis equal
    colorbar
    title('Source shape $\mathcal{X}$','interpreter','latex')

    subplot(1,3,3)
    hold off
    trisurf(Y.triv,Y.vert(:,1),Y.vert(:,2),Y.vert(:,3),weightsPlotY);
    axis equal
    colorbar
    title('Reference shape $\mathcal{Y}$','interpreter','latex')
    % 如果变形后的形状vertCurrFull的行数等于参考形状Y.vert的行数，则打印平均误差
    if size(vertCurrFull,1)==size(Y.vert,1)
        disp('mean error: ' + string(mean(normv(vertCurrFull-Y.vert))));
    end

    drawnow
end

%% createOutput
% 将结果封装为输出结构体，分别赋值给输出参数 POut、tauOut 和 COut
tauOut = tau;
COut = featOut.C;
POut = struct;
POut.assignment = featOut.assignment;
POut.assignmentinv = featOut.assignmentinv;

end
