clear;

T = zeros(10000,1);
T1 = true(10000,1);
T1(1:5000) = false;
T3 = false(10000,1);

%profile on
tic
T2 = T|T1;
toc
T3(1:500)=T(1:500)|T1(1:500);
toc
%profile viewer

