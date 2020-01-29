clear;
close all;
%%

% Generate F and G for just one objective

F1 = [zeros(3,3) eye(3);zeros(3,3) zeros(3,3)];
F2 = [zeros(3,3) eye(3);zeros(3,3) zeros(3,3)];

F = blkdiag(F1);

G1 = [zeros(3,3);eye(3,3)];
G2 = [zeros(3,3);eye(3,3)];
G = blkdiag(G1);

 %%
 
Q = eye(size(F));

[P,K,~,INFO] = icare(F,G,Q);

% P_Speed_Full = lyap(Fcl,QQ_Full);
P = round(P,5);
writematrix(P,'P.txt','Delimiter',' ')
writematrix(F,'F.txt','Delimiter',' ')
writematrix(G,'G.txt','Delimiter',' ')


% LfV_x = eta'*(F'*P+P*F)*eta;
% LgV_x = 2*eta'*P*G;
% V_x = eta'*P*eta;

% lambda_minQ = min(eig(QQ_Speed));
% lambda_maxP = max(eig(P1_Speed));

% lambda_minQ = min(eig(QQ_Pose));
% lambda_maxP = max(eig(P1_Pose));