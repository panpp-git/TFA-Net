function [Ts] = SST2(x,hlength);
% Computes the SST (Ts)  of the signal x.
% INPUT
%    x      :  Signal needed to be column vector.
%    hlength:  The hlength of window function.
% OUTPUT
%    Ts     :  The SST

[xrow,xcol] = size(x);
nfft=256;
if (xcol~=1),
 error('X must be column vector');
end; 

if (nargin < 1),
error('At least 1 parameter is required');
end;

if (nargin < 2),
hlength=round(xrow/5);
end;

Siglength=xrow;
hlength=hlength+1-rem(hlength,2);
ht = linspace(-0.5,0.5,hlength);ht=ht';

% Gaussian window
% h = exp(-pi/0.32^2*ht.^2);
h = kaiser(min(hlength,Siglength),10);
% derivative of window
% dh = -2*pi/0.32^2*ht .* h; % g'
dh=winderi(h,1);
[hrow,hcol]=size(h); Lh=(hrow-1)/2; 

N=xrow;
t=1:xrow;

[trow,tcol] = size(t);


tfr1= zeros (N,tcol) ; 
tfr2= zeros (N,tcol) ; 


Ts= zeros (nfft,tcol) ; 
% Pad the signal vector x
if rem(hlength,2)==1
    xp = [zeros((hlength-1)/2,1) ; x ; zeros((hlength-1)/2,1)];
else
    xp = [zeros((hlength)/2,1) ; x ; zeros((hlength-2)/2,1)];
end
xin = stft_rearrage(xp,length(xp),hlength,hlength-1,1);

% Compute the STFT
[tfr1,fout] = PerformDFT(bsxfun(@times,h,xin),nfft,1);
tfr2 = PerformDFT(bsxfun(@times,dh,xin),nfft,1);

m = floor(hlength/2);
inds = 0:nfft-1;
ez = exp(-1i*2*pi*m*inds/nfft)';
sstout = bsxfun(@times,tfr1,ez);

ft = 1:nfft;
bt = 1:N;

%%operator omega
nb = length(bt);
neta = length(ft);

omega = zeros (nfft,tcol);

fout=0:1/nfft:1-1/nfft;
for b=1:nb
    fcorr=-imag(tfr2(ft,b)./tfr1(ft,b));
    fcorr(~isfinite(fcorr)) = 0;
    omega(:,b) =1+ mod(round((fout'+fcorr)*(nfft-1)),nfft);
%         omega(:,b)=ft'-imag(tfr2(ft,b)./tfr1(ft,b));
end 
% omega=round(omega);

for b=1:nb%time
    % Reassignment step
    for eta=1:neta%frequency
%             if sstout(eta,b)>0.000001
                k = omega(eta,b);
                if k>=1 && k<=neta
                    Ts(k,b) = Ts(k,b) + sstout(eta,b);
                end
%             end
    end
end
Ts=Ts/(xrow/2);
end