% data = load('acc_mat_rbf.mat');
xvalues = d_vec;
yvalues = C_vec;
h = heatmap(yvalues, xvalues, acc_mat)
h.Title = 'How Polynomial Kernel Dimension and Soft vs Hard Margins Affect Model Accuracy';
h.XLabel = 'C Param - Hardness vs Softness';
h.YLabel = 'd (Polynomial Kernel Dimension)';