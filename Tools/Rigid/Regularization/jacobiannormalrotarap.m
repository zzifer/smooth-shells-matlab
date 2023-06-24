% 返回M和z
% M是一个雅可比矩阵，用于描述 ARAP（As-Rigid-As-Possible）变形在给定固定法线旋转下的变化情况。
% 它是一个大小为 [nSamples nSamples] 的矩阵，其中 nSamples 是样本数量。M 描述了样本之间的相互影响关系，可以用于进行形状变形和优化过程。
% z是一个增量向量，表示 ARAP 变形的结果。它是一个大小为 [nVert 1] 的向量，其中 nVert 是顶点的数量。
% z 表示每个顶点的变形增量，即变形后的顶点位置与参考顶点位置之间的差异
function [M,z] = jacobiannormalrotarap(vert,vertNew,normalNew,w,neigh,samples)
    % jacobiannormalrotarap - compute the arap jacobian for a fixed normal rotation
    %
    % vert: original vertices
    % vertNew: reference vertices
    % normalNew: morphed normals
    % w: SO3 lie algebra element of the normal rotation
    % neigh: connectivity information for vert
    % samples: samples of vert to compute the jacobian for

    neighRow = neigh.row;
    neighCol = neigh.col;
    
    normW = normv(w);
    L = normW<1e-10;
    normW(L) = 1e-10;
    
    nVertDiff = length(neighRow);
    nVert = size(vert,1);
    nSamples = length(samples);
    
    RwRow = rotationblock(w(neighRow,:),normW(neighRow),false);
     
    vertRow = vert(samples(neighRow),:);
    vertNewRow = vertNew(samples(neighRow),:);
    vertCol = vert(neighCol,:);
    vertNewCol = vertNew(neighCol,:);
    
    vertRow = vertRow';
    vertRow = vertRow(:);
    vertCol = vertCol';
    vertCol = vertCol(:);
    
    vertNewRow = vertNewRow';
    vertNewRow = vertNewRow(:);
    vertNewCol = vertNewCol';
    vertNewCol = vertNewCol(:);
    
    rotJac = RwRow*(vertCol-vertRow);
    rotJacHat = hatOpDiag(reshape(rotJac,3,nVertDiff)',false);
    

    normalNewBlock = constructblockmat(normalNew(neighRow,:),1:nVertDiff,1:nVertDiff,[nVertDiff nVertDiff])';
    rotJacHat = rotJacHat * normalNewBlock;
    
    
    z = blockstackmat(constructblockmat(rotJacHat' * (rotJac-(vertNewCol-vertNewRow)),neighRow,neighCol,[nSamples nVert]),nVert,1);
    
    M = constructblockmat(blockstackmat(constructblockmat(blockstackmat(rotJacHat' * rotJacHat,nVertDiff,1),neighRow,neighCol,[nSamples nVert]),nVert,1),1:nSamples,1:nSamples,[nSamples nSamples]);

end
