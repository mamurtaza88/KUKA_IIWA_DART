clear
close all
% save("z_axis.mat",'x_ee','x_t','t')

load('z_axis.mat')
x_ee1 = x_ee;
x_ee1(:,3) = x_ee1(:,3)-0.1;
figure(1)
hold on
plot(t,x_ee(:,3),'r');
plot(t,x_t(:,3),'b');
% plot(t,x_ee(:,2),'r.');
% plot(t,x_t(:,2),'b.');
% plot(t,x_ee(:,1),'r-');
% plot(t,x_t(:,1),'b-');
xlabel('time (s)');
ylabel('meter (m)');
legend('End-Effector','Target');
