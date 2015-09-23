k = 1:6;
rk = [0.031995701340888583, 0.085790000732726024, 0.17591529687614493, 0.19611410986004929, 0.24589062843465304, 0.27268397528271016]; 
Rk = [ 0.03299709, 0.08505727, 0.1749139, 0.19511272, 0.24662335, 0.2734167];

rkw = [0.038, 0.071, 0.15, 0.19, 0.24, 0.27]; 
Rkw = [ 0.039, 0.071, 0.154, 0.19, 0.24, 0.27];

plot(k,rk, '*r');
set(gca,'FontSize',12)
hold on;
plot(k,Rk, 'or');
plot(k,rkw, '*b');
plot(k,Rkw, 'ob');
xlabel('Annotator id: k');
ylabel('Estimates for r_k and the mean of R_k');
h=legend('r_k: Red wine dataset','Mean of R_k: Red wine dataset','r_k: White wine dataset','Mean of R_k: White wine dataset')
h.Location = 'northwest'
legend boxoff;
