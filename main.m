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




%Num_Data = numbers(:,[29:33, 35:49,60,61]);
% Num_Data = numbers(:,[29:33, 35:49]);%Absolute concentrations
% Num_Data = numbers(:,[60]);%PWI
%Num_Data = numbers(:,[61]);%CIA-K


Num_Data = numbers(:,[12,23,25,60,61,63:71]);


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

compare = 50;
scatter(Sf(:,1), Sf(:,2), 50, numbers(:,compare), 'filled');%50 is the size of the dot. So you can have to size based on a variable. The DATA term is color

plot([0 Ar(1,1)], [0 Ar(1,2)], 'm-', 'linewidth', 3)
plot([0 Ar(2,1)], [0 Ar(2,2)], 'c-', 'linewidth', 3) 
plot([0 Ar(3,1)], [0 Ar(3,2)], 'k-', 'linewidth', 3) 


%Percentage of Varience PoV
PoV = 100*diag(Lambda)/trace(Lambda);
fprintf('\nPercentage of Varience\n');
disp(PoV);


figure(2)

plot(Sf(:,1),numbers(:,compare), 'm*');
