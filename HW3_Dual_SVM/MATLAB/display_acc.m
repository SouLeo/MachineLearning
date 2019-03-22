data = load('acc_mat_rbf.mat');
xvalues = gamma_vec;
yvalues = C_vec;
h = heatmap(yvalues, xvalues, acc_mat)
h.Title = 'How RBF Kernel Size and Soft vs Hard Margins Affect Model Accuracy';
h.XLabel = 'C Param - Hardness vs Softness';
h.YLabel = 'Gamma (RBF Kernel Size)';