alpha = [1 1.1 1.2 1.3 1.4 1.5 1.6];

mv_rw = [95.9 92.8 90.53 87.26 86.61 85.56 83.97]; 
iam_rw = [55.2 53.6 54.26 53.30 49.30 51.45 50.76];
jam_rw = [97.9 95.75 92.92 89.74 88.23 86.69 84.38];
vrjam_rw = [98 95.81 92.98 89.74 88.23 86.69 84.76];


mv_ww = [96.1 91.64 87.14 84.42 84.14 82.92 82.05];
iam_ww = [55.33 55.13 51.91 52.66 52.74 52.75 51.88];
jam_ww = [97.94 94.91 89.88 86.80 85.4 83.85 82.65];
vrjam_ww = [97.99 94.97 89.89 86.80 85.4 83.85 82.68];

hold off;
plot (alpha,mv_rw,'--g*');
hold on;
%plot (alpha,iam_rw,'--bd');
plot (alpha,jam_rw,'--rx');
plot (alpha,vrjam_rw,'--ko');
ylabel('Accuracy in predicting the ground truth');
xlabel('\alpha','Interpreter','Tex');
%legend('MV','IAM','JAM','VRJAM');
legend('MV','JAM','VRJAM');
title('(a)');

figure;
plot (alpha,iam_rw,'--bd');
legend('IAM');
xlabel('\alpha','Interpreter','Tex');


figure;
hold off;
plot (alpha,mv_ww,'--g*');
hold on;
%plot (alpha,iam_ww,'--bd');
plot (alpha,jam_ww,'--rx');
plot (alpha,vrjam_ww,'--ko');
ylabel('Accuracy in predicting the ground truth');
xlabel('\alpha','Interpreter','Tex');
%legend('MV','IAM','JAM','VRJAM');
legend('MV','JAM','VRJAM');
title('(b)');

figure;
plot (alpha,iam_ww,'--bd');
legend('IAM');
xlabel('\alpha','Interpreter','Tex');
