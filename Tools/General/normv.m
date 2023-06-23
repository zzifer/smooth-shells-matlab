function nn = normv(v)
  % 计算向量 v 的每个元素的平方，并对每行进行求和
  % 然后，对每行的求和结果应用平方根函数 sqrt，得到欧氏范数
  nn = sqrt(sum(v.^2,2));
end
