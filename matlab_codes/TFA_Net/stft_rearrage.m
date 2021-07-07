function [xin,t] = stft_rearrage(x,nx,nwin,noverlap,Fs)
%getSTFTColumns re-orders input signal into matrix with overlap
%   This function is for internal use only. It may be removed in the future. 
%
%   Copyright 2016-2019 The MathWorks, Inc. 
%#codegen

% Determine the number of columns of the STFT output (i.e., the S output)
classCast = class(x); 
numChannels = size(x,2);
numSample = size(x,1);
ncol = fix((nx-noverlap)/(nwin-noverlap));
if ~isreal(x)
    xin = complex(zeros(nwin,ncol,numChannels,classCast)); 
else
    xin = zeros(nwin,ncol,numChannels,classCast); 
end

% Determine the number of columns of the STFT output (i.e., the S output)
coloffsets = (0:(ncol-1))*(nwin-noverlap);
rowindices = (1:nwin)';

% Segment x into individual columns with the proper offsets for each input
% channel
winPerCh = bsxfun(@plus,rowindices,coloffsets);
for iCh = 1:numChannels     
    xin(:,:,iCh) = x(winPerCh+(iCh-1)*numSample);
end

% Return time vector whose elements are centered in each segment
t = (coloffsets+(nwin/2)')/Fs;