function Wdt = winderi(w,Fs)
%DTWIN differentiate window in time domain via cubic spline interpolation
%   This function is for internal use only. It may be removed in the future. 
%   
%   See also DFWIN.
%
%   Copyright 2016-2018 The MathWorks, Inc.

%#codegen

% compute the piecewise polynomial representation of the window
% and fetch the coefficients
n = numel(w);
pp = spline(1:n,w);

% take the derivative of each polynomial and evaluate it over the same
% samples as the original window

if coder.target('MATLAB')
    [breaks,coefs,npieces,order,dim] = unmkpp(pp);
    ppd = mkpp(breaks,repmat(order-1:-1:1,dim*npieces,1).*coefs(:,1:order-1),dim);
else
    ppd = coder.internal.ppder(pp);
end

Wdt = ppval(ppd,(1:n)').*(Fs/(2*pi));

% LocalWords:  DFWIN
