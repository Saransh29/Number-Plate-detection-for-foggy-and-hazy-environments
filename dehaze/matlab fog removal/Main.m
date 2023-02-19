clc;
close all;
clear all;

% input an image
filename = '.\examples\sam_6';
HazeImg = imread([filename, '.bmp']);
figure, imshow(HazeImg, []);

% estimating the global airlight
% method = 'our'; 
% method = 'he'; 
method = 'manual'; 
wsz = 15; % window size
A = Airlight(HazeImg, method, wsz); 

% calculating boundary constraints
wsz = 3; % window size
ts = Boundcon(HazeImg, A, 30, 300, wsz);

% refining the estimation of transmission
lambda = 2;  % regularization parameter, the more this parameter, the closer to the original patch-wise transmission
t = CalTransmission(HazeImg, ts, lambda, 0.5); % using contextual information

% dehazing
r = Dehazefun(HazeImg, t, A, 0.85); 

% show and save the results
figure, imshow(ts, []);
figure, imshow(1-t, []); colormap hot;
figure, imshow(r, []);

