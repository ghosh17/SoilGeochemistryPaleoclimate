%%

%%clear, close, clc all function
clear all
close all
clc

%How many minerals (that contaain common metals) are there in soils?

%Define soil type from 1-11, 1 being inceptisol and 11 being Oxisol. Order
%determined by time required for soil formation.

[numbers, strings, raw] = xlsread('Data.xlsx');

%Eleminate unsuitable samples

[I] = find(isnan(sum(numbers(:,[12, 29:39]),2)));

numbers(I, :)= [];

numbers(numbers(:, 12)== -99, :)= [];%Delete samples with empty soil type
numbers(numbers(:, 65)== 1, :)= [];%Delete soils that have been distrubed by human activity

% %tweek soil type parameters
% numbers(numbers(:, 12)~= 10, :)= [];%Delete samples with empty soil type
% 
% %Soil parent material
% %Sedimentary protolith only
% numbers(numbers(:, 25)== -99, :)= [];
% numbers(numbers(:, 25)== 2, :)= [];

% numbers = numbers([1:5, 87:346, 348:449, 540:811, 826:842],:); %Excluding rows that have spotty data & human disturbed cultivated soils.
% numbers = numbers([1:405, 407:412],:);
% %If elements are missing or soil has been disturbed delete row
% 

%How do I remove all NaNs in a column
%numbers(isnan(numbers(:, [41:49])),:) = [] ;


%TODO: If soil type is inceptisol, entisol or ardisol, sopodosols remove

% numbers(numbers(:, 12)== 1, :)= [];
% numbers(numbers(:, 12)== 2, :)= [];
% numbers(numbers(:, 12)== 3, :)= [];
% numbers(numbers(:, 12)== 4, :)= [];

%Remove empty elements
%numbers(numbers(:, 41)== NaN, :)= [];



%Num_Data = numbers(:,[29:33, 35:49,60,61]);
% Num_Data = numbers(:,[29:33, 35:49]);%Absolute concentrations
% Num_Data = numbers(:,[60]);%PWI
%Num_Data = numbers(:,[61]);%CIA-K



%%How many minerals that contain major metal elements are present in soil?

Num_Minerals = numbers(:,[29:39]); %remove Si becasue it is ubiquitous to all clay minerals

%Step 1: Column Standardize
DATA_MINERALS = colstd(Num_Minerals);

%Step 2: Covarience Matrix
R_MINERALS = cov(DATA_MINERALS);

%Step 3: Eigen vector and value
[V_MINERALS,Lambda_MINERALS] = eig(R_MINERALS);

%Lambda sorted wrong way
lambda_MINERALS = diag(Lambda_MINERALS);
[I_MINERALS,J_MINERALS] = sort(lambda_MINERALS, 'descend');
V_MINERALS = V_MINERALS(:,J_MINERALS);
%Remake the big Lambda
Lambda_MINERALS = diag(lambda_MINERALS(J_MINERALS));

%V contains eigenvectors as it's columns and Lambda contains eigenvalues on
%its diagonals 
% fprintf('\nEigenvectors are \n');
% disp(V_MINERALS)

fprintf('\nRelative abundance of minerals i.e. eigenvalues =\n');
disp(diag(Lambda_MINERALS));


%Step 4: New principle components. Factor loadings for each variable on each component Ar = V.S or V.root(lambda); Sr = Data.V
Ar_MINERALS = V_MINERALS*sqrt(Lambda_MINERALS);

Sr_MINERALS = DATA_MINERALS*V_MINERALS; %Principle component score: Projection of each data vector onto new component axes

Sf_MINERALS = DATA_MINERALS*Ar_MINERALS; %Factor score:Same as principle component score only that it has been scaled by the magnitude of singular vector

fprintf('\nFactor loading i.e. trace element compositions:\n');
disp(abs(Ar_MINERALS));

figure(1)

plot(Sf_MINERALS(:,1), Sf_MINERALS(:,2), 'o')
hold on
xlabel('PC1')
ylabel('PC2')


compare = 12;

scatter(Sf_MINERALS(:,1), Sf_MINERALS(:,2), 50, numbers(:,12), 'filled');
set(gca, 'Color',[1 0.92 0.8])


plot([0 Ar_MINERALS(1,1)], [0 Ar_MINERALS(1,2)], 'r-', 'linewidth', 3)%Metal 1 
plot([0 Ar_MINERALS(2,1)], [0 Ar_MINERALS(2,2)], 'k-', 'linewidth', 3)%Metal 2
plot([0 Ar_MINERALS(3,1)], [0 Ar_MINERALS(3,2)], 'm-', 'linewidth', 3)%Metal 3
plot([0 Ar_MINERALS(4,1)], [0 Ar_MINERALS(4,2)], 'c-', 'linewidth', 3)%Metal 4
plot([0 Ar_MINERALS(5,1)], [0 Ar_MINERALS(5,2)], 'b-', 'linewidth', 3)%Metal 5
plot([0 Ar_MINERALS(6,1)], [0 Ar_MINERALS(6,2)], 'y-', 'linewidth', 3)%Metal 6


%Percentage of Varience PoV
PoV_MINERALS = 100*diag(Lambda_MINERALS)/trace(Lambda_MINERALS);
fprintf('\nPercentage of Varience\n');
disp(PoV_MINERALS);


%Factor Analysis (FA) statistical extension of PCA where you discard some
%of the less significant principle components. New covariience matrix
%retained factors will be approximation. ie retained contains signal and
%discarded contains noise

%Lets select the first three components 
sig=3;
AR_MINERALS = Ar_MINERALS(:,1:sig);
tmp_MINERALS=AR_MINERALS.^2;
h_sq_MINERALS = sum(tmp_MINERALS,2);

fprintf('\nCommunalities: How much of the original varience was contained in variable using %f PC\n',sig);
disp(h_sq_MINERALS);



figure(2)
h1=plot(abs(Ar_MINERALS(:,1)),'o-'); hold on
h2=plot(abs(Ar_MINERALS(:,2)),'^-');
h3=plot(abs(Ar_MINERALS(:,3)),'s-');
h4=plot(abs(Ar_MINERALS(:,4)),'*-');
h5=plot(abs(Ar_MINERALS(:,5)),'*-');
h6=plot(abs(Ar_MINERALS(:,6)),'*-');
h7=plot(abs(Ar_MINERALS(:,7)),'*-');
h8=plot(abs(Ar_MINERALS(:,8)),'*-');
h9=plot(abs(Ar_MINERALS(:,9)),'*-');
set(gca, 'xtick', [1 2 3 4 5 6 7 8 9])
set(gca, 'xticklabel', [{'S1'} {'S2'} {'S3'} {'S4'} {'S5'} {'S6'} {'S7'} {'S8'} {'S9'}])
legend([h1 h2 h3 h4 h5 h6 h7 h8 h9], 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9')




Fe = numbers(:,29);
Mn = numbers(:,30);
P = numbers(:,31);
Ti = numbers(:,33);
Al = numbers(:,35);
Ca = numbers(:,36);
Na = numbers(:,37);
Mg = numbers(:,38);
K = numbers(:,39);

MAP = numbers(:,50);
GSP = numbers(:,51);
SP = numbers(:,54);

MAT = numbers(:,55);
GST = numbers(:,56);
ST = numbers(:,59);

PWI = numbers(:,60);
CIA = numbers(:,61);
PWI_temp = -2.74*log(PWI) + 21.39;%PWI temperature correction
CIA_precip = 221.1*(exp(0.0197*CIA)); 


soil_type = numbers(:,12);


figure(3)
scatter(Sf_MINERALS(:,1), Al, 50, numbers(:,12), 'filled');
title('PC1 v/s [Al_2O_3]');
xlabel('PC1');
ylabel('Al_2O_3');

figure(4)
scatter(Sf_MINERALS(:,2), Al, 50, numbers(:,12), 'filled');
title('PC2 v/s [Al_2O_3]');
xlabel('PC1');
ylabel('Al_2O_3'); 

figure(5)
scatter(Sf_MINERALS(:,1), K, 50, numbers(:,12), 'filled');
title('PC1 v/s [K_2O]');
xlabel('PC1');
ylabel('K_2O'); 


figure(6)
scatter(Sf_MINERALS(:,2), K, 50, numbers(:,12), 'filled');
title('PC2 v/s [K_2O]');
xlabel('PC1');
ylabel('K_2O'); 

figure(7)
scatter(Sf_MINERALS(:,1), Mg, 50, numbers(:,12), 'filled');
title('PC1 v/s [MgO]');
xlabel('PC1');
ylabel('MgO'); 

figure(8)
scatter(Sf_MINERALS(:,2), Mg, 50, numbers(:,12), 'filled');
title('PC2 v/s [MgO]');
xlabel('PC2');
ylabel('MgO'); 


figure(9)
scatter(Sf_MINERALS(:,1), Fe, 50, numbers(:,12), 'filled');
title('PC1 v/s [Fe_2O_3]');
xlabel('PC1');
ylabel('Fe_2O_3'); 

figure(10)
scatter(Sf_MINERALS(:,2), Fe, 50, numbers(:,12), 'filled');
title('PC2 v/s [Fe_2O_3]');
xlabel('PC1');
ylabel('Fe_2O_3'); 

figure(11)
scatter(Sf_MINERALS(:,1), Ca, 50, numbers(:,12), 'filled');
title('PC1 v/s [CaO]');
xlabel('PC1');
ylabel('CaO'); 

figure(12)
scatter(Sf_MINERALS(:,2), Ca, 50, numbers(:,12), 'filled');
title('PC2 v/s [CaO]');
xlabel('PC2');
ylabel('CaO');

figure(13)
scatter(Sf_MINERALS(:,1), Ti, 50, numbers(:,12), 'filled');
title('PC1 v/s [TiO]');
xlabel('PC1');
ylabel('TiO'); 

figure(14)
scatter(Sf_MINERALS(:,2), Ti, 50, numbers(:,12), 'filled');
title('PC2 v/s [TiO]');
xlabel('PC2');
ylabel('TiO');

figure(15)
scatter(Sf_MINERALS(:,1), Na, 50, numbers(:,12), 'filled');
title('PC1 v/s [Na_2O]');
xlabel('PC1');
ylabel('Na_2O'); 

figure(16)
scatter(Sf_MINERALS(:,2), Na, 50, numbers(:,12), 'filled');
title('PC2 v/s [Na_2O]');
xlabel('PC2');
ylabel('Na_2O');




%PC1 = a[Al] + b[Fe] - c[K] -d[Mg];

PC1 = Sf_MINERALS(:,1);
PC2 = Sf_MINERALS(:,2);
PC3 = Sf_MINERALS(:,2);

%geochem_matrix = [Al Fe K Mg];
geochem_matrix = [Ti Fe Al Mn Ca Mg K Na P];
Weight_term1 = PC1 \ geochem_matrix;
Weight_term2 = PC2 \ geochem_matrix;
Weight_term3 = PC3 \ geochem_matrix;

SGI_matrix_PC1 = [Ti Fe Al];
SGI_matrix_PC2 = [Ca Na Mg];

SGI_Weight_term1 = PC1 \ SGI_matrix_PC1;
SGI_Weight_term2 = PC2 \ SGI_matrix_PC2;

SCI =  SGI_Weight_term1(1) .* Ti + SGI_Weight_term1(2) .* Fe + SGI_Weight_term1(3) .* Al + SGI_Weight_term2(1) .* Ca + SGI_Weight_term2(2) .* Na + SGI_Weight_term2(3) .* Mg;

predicted_soil1 = ( Weight_term1(1) .* Ti + Weight_term1(2) .* Fe + Weight_term1(3) .* Al + Weight_term1(4) .* Mn + Weight_term1(5) .* Ca + Weight_term1(6) .* Mg + Weight_term1(7) .* K + Weight_term1(8) .* Na + Weight_term1(9) .* P);

predicted_soil2 = ( Weight_term2(1) .* Ti + Weight_term2(2) .* Fe + Weight_term2(3) .* Al + Weight_term2(4) .* Mn + Weight_term2(5) .* Ca + Weight_term2(6) .* Mg + Weight_term2(7) .* K + Weight_term2(8) .* Na + Weight_term2(9) .* P);

predicted_soil3 = ( Weight_term3(1) .* Ti + Weight_term3(2) .* Fe + Weight_term3(3) .* Al + Weight_term3(4) .* Mn + Weight_term3(5) .* Ca + Weight_term3(6) .* Mg + Weight_term3(7) .* K + Weight_term3(8) .* Na + Weight_term3(9) .* P);

%Age_index = log(Al.*Fe./K);

soil_form_age = [10e2, 10e3, 5*10e4, 5*10e4, 5*10e4, 5*10e5, 10e6]; % Vertisol(5)->10e2, Mollisol(6)->10e3,Aridisol(7)-> 5*10e4 Alfisol(8)-> 5*10e4, Aridisol(9)-> 5*10e4 ,10-> 5*10e5 ,11-> 10e6



figure(100)
subplot(3,1,1)
scatter(numbers(:,12), predicted_soil1, 50, numbers(:,12), 'filled');
xlabel('Soil type');
ylabel('Predicted soil geochemistry from PC1');
subplot(3,1,2)
scatter(numbers(:,12), PWI, 50, numbers(:,12), 'filled');
xlabel('Soil type');
ylabel('PWI');
subplot(3,1,3)
scatter(numbers(:,12), SCI, 50, numbers(:,12), 'filled');
xlabel('Soil type');
ylabel('Predicted soil geochemistry index');

% scatter(numbers(:,25), predicted_soil, 50, numbers(:,25), 'filled');
% xlabel('Parent Material');
% ylabel('Predicted soil geochemistry index');

%PC1- Geochem indices
figure(101)
subplot(3,1,1)
title('New soil geochemistry v/s PC1');
scatter(predicted_soil1, Sf_MINERALS(:,1), 50, numbers(:,12), 'filled');
xlabel('PC1');
ylabel('New Index');

subplot(3,1,2)
title('CIA-K v/s PC1');
scatter(CIA, Sf_MINERALS(:,1), 50, numbers(:,12), 'filled');
xlabel('PC1');
ylabel('CIA');

subplot(3,1,3)
title('PWI v/s PC1');
scatter(PWI, Sf_MINERALS(:,1), 50, numbers(:,12), 'filled');
xlabel('PC1');
ylabel('PWI');

%PC1- Climate
figure(102)
subplot(4,1,1)
title('PC1 v/s MAP');
scatter(MAP, Sf_MINERALS(:,1), 50, numbers(:,12), 'filled');
xlabel('PC1');
ylabel('MAP');

subplot(4,1,2)
title('PC1 v/s Growing season precipitation');
scatter(GSP, Sf_MINERALS(:,1), 50, numbers(:,12), 'filled');
xlabel('PC1');
ylabel('GSP');

subplot(4,1,3)
title('PC1 v/s MAT');
scatter(MAT, Sf_MINERALS(:,1), 50, numbers(:,12), 'filled');
xlabel('PC1');
ylabel('MAT');

subplot(4,1,4)
title('PC1 v/s Growing season precipitation');
scatter( GST, Sf_MINERALS(:,1), 50, numbers(:,12), 'filled');
xlabel('PC1');
ylabel('GST');

%PC2- Geochem indices
figure(103)
subplot(3,1,1)
title('New soil geochemistry v/s PC1');
scatter(predicted_soil2, PC2, 50, numbers(:,12), 'filled');
xlabel('PC2');
ylabel('New Index');

subplot(3,1,2)
title('CIA-K v/s PC2');
scatter(CIA, PC2, 50, numbers(:,12), 'filled');
xlabel('PC2');
ylabel('CIA');

subplot(3,1,3)
title('PWI v/s PC2');
scatter(PWI, PC2, 50, numbers(:,12), 'filled');
xlabel('PC2');
ylabel('PWI');

%PC2- Climate
figure(104)
subplot(4,1,1)
title('PC2 v/s MAP');
scatter(MAP, PC2, 50, numbers(:,12), 'filled');
xlabel('PC2');
ylabel('MAP');

subplot(4,1,2)
title('PC2 v/s Growing season precipitation');
scatter(GSP, PC2, 50, numbers(:,12), 'filled');
xlabel('PC2');
ylabel('GSP');

subplot(4,1,3)
title('PC2 v/s MAT');
scatter(MAT, PC2, 50, numbers(:,12), 'filled');
xlabel('PC2');
ylabel('MAT');

subplot(4,1,4)
title('PC2 v/s Growing season precipitation');
scatter(GST, PC2, 50, numbers(:,12), 'filled');
xlabel('PC2');
ylabel('GST');

%MAP
figure(105)
subplot(3,1,1)
scatter(MAP, predicted_soil1, 50, numbers(:,12), 'filled');
xlabel('MAP');
ylabel('Index_1');


subplot(3,1,2)
scatter(MAP, CIA_precip, 50, numbers(:,12), 'filled');
xlabel('MAP');
ylabel('CIA');

subplot(3,1,3)
scatter(MAP, PWI_temp, 50, numbers(:,12), 'filled');
xlabel('MAP');
ylabel('PWI');

%MAT
figure(106)
subplot(3,1,1)
scatter(MAT, predicted_soil1, 50, numbers(:,12), 'filled');
xlabel('Index_1');
ylabel('MAT');

subplot(3,1,2)
scatter(MAT, CIA_precip, 50, numbers(:,12), 'filled');
xlabel('MAT');
ylabel('CIA');


subplot(3,1,3)
scatter(MAT, PWI_temp,  50, numbers(:,12), 'filled');
xlabel('MAT');
ylabel('PWI');




%GSP-> Growing season precipitation
figure(107)
subplot(3,1,1)
scatter(GSP, predicted_soil1, 50, numbers(:,12), 'filled');
xlabel('Growing season precipitation');
ylabel('My Predicted index');
subplot(3,1,2)
scatter(GSP, CIA_precip, 50, numbers(:,12), 'filled');
xlabel('Growing season precipitation');
ylabel('CIA');
subplot(3,1,3)
scatter(GSP, PWI_temp, 50, numbers(:,12), 'filled');
xlabel('Growing season precipitation');
ylabel('PWI');

%SP -> Summer precipitation
figure(108)
subplot(3,1,1)
scatter(SP, predicted_soil1, 50, numbers(:,12), 'filled');
xlabel('Summer precipitation');
ylabel('My Predicted index');
subplot(3,1,2)
scatter(SP, CIA_precip, 50, numbers(:,12), 'filled');
xlabel('Summer precipitation');
ylabel('CIA');
subplot(3,1,3)
scatter(SP, PWI_temp, 50, numbers(:,12), 'filled');
xlabel('Summer precipitation');
ylabel('PWI');

%GST-> Growing season temperature
figure(109)
subplot(3,1,1)
scatter(GST, predicted_soil1, 50, numbers(:,12), 'filled');
xlabel('Growing season temperature');
ylabel('My Predicted index');
subplot(3,1,2)
scatter(GST, CIA_precip, 50, numbers(:,12), 'filled');
xlabel('Growing season temperature');
ylabel('CIA');
subplot(3,1,3)
scatter(GST, PWI_temp, 50, numbers(:,12), 'filled');
xlabel('Growing season temperature');
ylabel('PWI');

%ST -> Summer temperature
figure(110)
subplot(3,1,1)
scatter(ST, predicted_soil1, 50, numbers(:,12), 'filled');
xlabel('Summer temperature');
ylabel('My Predicted index');
subplot(3,1,2)
scatter(ST, CIA_precip, 50, numbers(:,12), 'filled');
xlabel('Summer temperature');
ylabel('CIA');
subplot(3,1,3)
scatter(ST, PWI_temp, 50, numbers(:,12), 'filled');
xlabel('Summer temperature');
ylabel('PWI');

figure(111)

subplot(4,1,1)
title('PC3 v/s MAP');
scatter(MAP, PC3, 50, numbers(:,12), 'filled');
xlabel('PC3');
ylabel('MAP');

subplot(4,1,2)
title('PC3 v/s Growing season precipitation');
scatter(GSP, PC3, 50, numbers(:,12), 'filled');
xlabel('PC3');
ylabel('GSP');

subplot(4,1,3)
title('PC3 v/s MAT');
scatter(MAT, PC3, 50, numbers(:,12), 'filled');
xlabel('PC3');
ylabel('MAT');

subplot(4,1,4)
title('PC3 v/s Growing season precipitation');
scatter(GST, PC3, 50, numbers(:,12), 'filled');
xlabel('PC3');
ylabel('GST');

%SCI- Precip
figure(112)
subplot(2,1,1)
title('MAP v/s SCI');
scatter(MAP, SCI, 50, numbers(:,12), 'filled');
xlabel('MAP');
ylabel('SCI');

subplot(2,1,2)
title('Growing season precipitation v/s SCI');
scatter(GSP, SCI, 50, numbers(:,12), 'filled');
xlabel('GSP');
ylabel('SCI');

%SCI - Temp
figure(113)

subplot(2,1,1)
title('MAT v/s SCI');
scatter(MAT, SCI, 50, numbers(:,12), 'filled');
xlabel('MAT');
ylabel('SCI');

subplot(2,1,2)
title('Growing season precipitation v/s SCI');
scatter( GST, SCI, 50, numbers(:,12), 'filled');
xlabel('GST');
ylabel('SCI');

%boxchart

figure(114)
boxchart(soil_type,SCI)
xlabel('Soil Type')
ylabel('SCI')

xticks([1 2 3 4 5 6 7 8 9 10 11])
xticklabels({'Inceptisols','Andisols', 'Entisol','Spodosols','Vertisol','Mollisol','Aridisol','Alfisol', 'Histosols', 'Ultisol', 'Oxisols'})


figure(115)
boxchart(soil_type,PWI)
xlabel('Soil Type')
ylabel('PWI')

xticks([1 2 3 4 5 6 7 8 9 10 11])
xticklabels({'Inceptisols','Andisols', 'Entisol','Spodosols','Vertisol','Mollisol','Aridisol','Alfisol', 'Histosols', 'Ultisol', 'Oxisols'})


figure(116)
boxchart(soil_type,CIA)
xlabel('Soil Type')
ylabel('CIA-K')

xticks([1 2 3 4 5 6 7 8 9 10 11])
xticklabels({'Inceptisols','Andisols', 'Entisol','Spodosols','Vertisol','Mollisol','Aridisol','Alfisol', 'Histosols', 'Ultisol', 'Oxisols'})



%TODO : Stepwise linear regression


%{

%%Find any relation between paleoclimatic parameters and soil Geochemistry 

%%clear, close, clc all function
clear all
close all
clc

%Define soil type from 1-11, 1 being inceptisol and 11 being Oxisol. Order
%determined by time required for soil formation.

[numbers, strings, raw] = xlsread('Data.xlsx');

%Eleminate unsuitable samples

numbers = numbers([1:5, 87:449, 540:811, 826:842],:); %Excluding rows that have spotty data & human disturbed cultivated soils.

%If elements are missing or soil has been disturbed delete row

numbers(numbers(:, 12)== -99, :)= [];%Delete samples with empty soil type
numbers(numbers(:, 23)== -9999, :)= [];%Discard if elevation unknown
numbers(numbers(:, 25)== -99, :)= [];%Delete samples with empty parent rock type

%If Precipitation and temperature data missing
numbers(numbers(:, 51)== 1, :)= [];
numbers(numbers(:, 52)== 1, :)= [];
numbers(numbers(:, 56)== 1, :)= [];
numbers(numbers(:, 57)== 1, :)= [];


% %Remove types of soils
% numbers(numbers(:, 12)== 6, :)= [];
% numbers(numbers(:, 12)== 7, :)= [];

Num_Data = numbers(:,[12, 23, 25, 63:71]);

%Perform PCA analysis
%colstd function from the 'Data Modeling for Ocean Sciences' textbook example code
%Step 1: Column Standardize

DATA = colstd(Num_Data); 

FULL_DATA = colstd(numbers);%

%Step 2: Covarience Matrix
R = cov(DATA);

%Step 3: Eigen vector and value
[V,Lambda] = eig(R);
%Lambda sorted wrong way
lambda = diag(Lambda);
[I,J] = sort(lambda, 'descend');
V = V(:,J);
%Remake the big Lambda
Lambda = diag(lambda(J));

Ar = V*sqrt(Lambda);

Sr = DATA*V; %Principle component score: Projection of each data vector onto new component axes

Sf = DATA*Ar; %Factor score:Same as principle component score only that it has been scaled by the magnitude of singular vector

figure(1)

plot(Sf(:,1), Sf(:,2), 'o')
hold on
xlabel('PC1')
ylabel('PC2')

compare = 12;

scatter(Sf(:,1), Sf(:,2), 50, numbers(:,compare), 'filled');%50 is the size of the dot. So you can have to size based on a variable. The DATA term is color

plot([0 Ar(1,1)], [0 Ar(1,2)], 'm-', 'linewidth', 3)
plot([0 Ar(2,1)], [0 Ar(2,2)], 'c-', 'linewidth', 3) 
plot([0 Ar(3,1)], [0 Ar(3,2)], 'k-', 'linewidth', 3) 


%Percentage of Varience PoV
PoV = 100*diag(Lambda)/trace(Lambda);
fprintf('\nPercentage of Varience\n');
disp(PoV);

fprintf('\nFactor loading i.e. trace element compositions:\n');
disp(abs(Ar));

figure(2)

scatter(Sf(:,1),numbers(:,61), 50, numbers(:,12), 'filled');
title('PC1 v/s CIA-K');
xlabel('PC1')
ylabel('CIA-K')

figure(3)

scatter(Sf(:,1),numbers(:,60), 50, numbers(:,12), 'filled');
title('PC1 v/s PWI');
xlabel('PC1')
ylabel('PWI')

figure(4)

scatter(Sf(:,2),numbers(:,61), 50, numbers(:,12), 'filled');
title('PC2 v/s CIA-K');
xlabel('PC2')
ylabel('CIA-K')

figure(5)

scatter(Sf(:,2),numbers(:,60), 50, numbers(:,12), 'filled');
title('PC2 v/s PWI');
xlabel('PC2')
ylabel('PWI')

ppt_start = 50;

figure(6)
plot(Sf(:,1),numbers(:,ppt_start), 'b*');
title('PC1 v/s WClimMAP');
xlabel('PC1')
ylabel('WClimMAP')

figure(7)
plot(Sf(:,1),numbers(:,(ppt_start+1)), 'bo');
title('PC1 v/s GSP5');
xlabel('PC1')
ylabel('GSP5')

figure(8)
plot(Sf(:,1),numbers(:,(ppt_start+2)), 'bo');
title('PC1 v/s GSP10');
xlabel('PC1')
ylabel('GSP10')

figure(9)
plot(Sf(:,1),numbers(:,(ppt_start+3)), 'bo');
title('PC1 v/s SP08');
xlabel('PC1')
ylabel('SP08')

figure(10)
plot(Sf(:,1),numbers(:,(ppt_start+4)), 'bo');
title('PC1 v/s SP09');
xlabel('PC1')
ylabel('SP09')

ppt_seasonality = abs(numbers(:,(ppt_start+2)) - numbers(:,(ppt_start+1)));

figure(11)
scatter(Sf(:,1),ppt_seasonality, 50, numbers(:,55), 'filled');
title('PC1 v/s precipitation seasonality w/ temperature shaded');
xlabel('PC1')
ylabel('abs(GSP5-GSP10)')

figure(12)
scatter(Sf(:,2),ppt_seasonality, 50, numbers(:,55), 'filled');
title('PC2 v/s precipitation seasonality w/ temperature shaded');
xlabel('PC2')
ylabel('abs(GSP5-GSP10)')

figure(13)
scatter(Sf(:,3),ppt_seasonality, 50, numbers(:,55), 'filled');
title('PC3 v/s precipitation seasonality w/ temperature shaded');
xlabel('PC3')
ylabel('abs(GSP5-GSP10)')


temp_start = 55;

figure(14)
plot(Sf(:,1),numbers(:,temp_start), 'r*');
title('PC1 v/s WClimMAT');
xlabel('PC1')
ylabel('WClimMAT')


figure(15)
plot(Sf(:,1),numbers(:,(temp_start+1)), 'ro');
title('PC1 v/s GST5');
xlabel('PC1')
ylabel('GST5')

figure(16)
plot(Sf(:,1),numbers(:,(temp_start+2)), 'ro');
title('PC1 v/s GST10');
xlabel('PC1')
ylabel('GST10')

figure(17)
plot(Sf(:,1),numbers(:,(temp_start+3)), 'ro');
title('PC1 v/s ST08');
xlabel('PC1')
ylabel('ST08')

figure(18)
plot(Sf(:,1),numbers(:,(temp_start+4)), 'ro');
title('PC1 v/s ST09');
xlabel('PC1')
ylabel('ST09')


temp_seasonality = abs(numbers(:,(temp_start+2)) - numbers(:,(temp_start+1)));

figure(19)
scatter(Sf(:,1),temp_seasonality, 50, numbers(:,50), 'filled');
title('PC1 v/s temperature seasonlity with precipitation shaded');
xlabel('PC1')
ylabel('abs(GST5-GST10)')

figure(20)
scatter(Sf(:,2),temp_seasonality, 50, numbers(:,50), 'filled');
title('PC2 v/s temperature seasonlity with precipitation shaded');
xlabel('PC2')
ylabel('abs(GST5-GST10)')

figure(21)
scatter(Sf(:,3),temp_seasonality, 50, numbers(:,50), 'filled');
title('PC3 v/s temperature seasonlity with precipitation shaded');
xlabel('PC3')
ylabel('abs(GST5-GST10)')

my_seasonality_index = colstd(temp_seasonality) .* colstd(ppt_seasonality);

figure(22)
scatter(Sf(:,1),my_seasonality_index, 50, numbers(:,12), 'filled');
title('PC1 v/s seasonality of temperature and precipitation');
xlabel('PC1')
ylabel('Seasonality')

figure(23)
scatter(Sf(:,2),my_seasonality_index, 50, numbers(:,12), 'filled');
title('PC2 v/s seasonality of temperature and precipitation');
xlabel('PC2')
ylabel('Seasonality')

figure(24)
scatter(Sf(:,3),my_seasonality_index, 50, numbers(:,12), 'filled');
title('PC3 v/s seasonality of temperature and precipitation');
xlabel('PC3')
ylabel('Seasonality')

figure(25)
scatter(temp_seasonality, numbers(:,60), 50, numbers(:,12), 'filled');
title('Seasonality v/s PWI');
xlabel('PWI')
ylabel('Seasonality')
%}
