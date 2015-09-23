k = [3 5 7 9];

mv_rw = [89.12 90.27 91.72 93.50] ; 
iam_rw = [55.52 51.97 52.94 52.57]; 
jam_rw = [89.12 91.13 92.68 94.24];
vrjam_rw = [89.12 91.90 92.59 94.38]; 

mv_ww = [87.19 88.82 89.78 91.64]; 
iam_ww = [57.77 55.19 53.48 52.81]; 
jam_ww = [87.19 89.09 90.08 92.61];
vrjam_ww = [87.19 89.09 90.08 92.61];

figure; 
hold off;
plot (k,mv_rw,'--g*');
hold on;
%plot (k,iam_rw,'--bd');
plot (k,jam_rw,'--rx');
plot (k,vrjam_rw,'--ko');
ylabel('Accuracy in predicting the ground truth');
xlabel('k','Interpreter','Tex');
set(gca,'XTick',[3 5 7 9])
%legend('MV','IAM','JAM','VRJAM');
legend('MV','JAM','VRJAM');
title('(a)');

figure;
plot (k,iam_rw,'--bd');
legend('IAM');
xlabel('k','Interpreter','Tex');
set(gca,'XTick',[3 5 7 9])


figure;
hold off;
plot (k,mv_ww,'--g*');
hold on;
%plot (k,iam_ww,'--bd');
plot (k,jam_ww,'--rx');
plot (k,vrjam_ww,'--ko');
ylabel('Accuracy in predicting the ground truth');
xlabel('k','Interpreter','Tex');
set(gca,'XTick',[3 5 7 9])
%legend('MV','IAM','JAM','VRJAM');
legend('MV','JAM','VRJAM');
title('(b)');

figure;
plot (k,iam_ww,'--bd');
legend('IAM');
xlabel('k','Interpreter','Tex');
set(gca,'XTick',[3 5 7 9])
