function [Te,IF] = SET_Y2(x,hlength);
%   Synchroextracting Transform
%	x       : Signal.
%	hlength : Window length.

%   IF   : Synchroextracting operator representation.
%   Te   : SET result.
%   tfr  : STFT result
%  This program is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
%
%   Written by YuGang in Shandong University at 2016.5.13.
nfft=128;
[xrow,xcol] = size(x);

N=xrow;

if (xcol~=1),
 error('X must be column vector');
end;

if (nargin < 2),
 hlength=round(xrow/8);
end;

t=1:N;
[trow,tcol] = size(t);

hlength=hlength+1-rem(hlength,2);
ht = linspace(-0.5,0.5,hlength);ht=ht';

% Gaussian window
% h = exp(-pi/0.32^2*ht.^2);
h = kaiser(min(hlength,N),10);
% derivative of window
% dh = -2*pi/0.32^2*ht .* h; % g'
dh=winderi(h,1);
[hrow,hcol]=size(h); Lh=(hrow-1)/2; 

N=xrow;
t=1:xrow;

[trow,tcol] = size(t);


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

omega = zeros (nfft,tcol);
fout=0:1/nfft:1-1/nfft;
ft=1:nfft;
for b=1:tcol
    fcorr=-imag(tfr2(ft,b)./tfr1(ft,b));
    fcorr(~isfinite(fcorr)) = 0;
    omega(:,b) =1+ mod(round((fout'+fcorr)*(nfft-1)),nfft);
%         omega(:,b)=ft'-imag(tfr2(ft,b)./tfr1(ft,b));
end 
E=mean(abs(x));
IF=zeros(nfft,N);
for i=1:nfft%frequency
for j=1:N%time
%      if abs(tfr1(i,j))>0.8*E%if you are interested in weak signals, you can delete this line.
         %if abs(1-real(va*1i*tfr2(i,j)/2/pi./tfr1(i,j)))<0.5
         if abs(omega(i,j)-i)<0.5
            IF(i,j)=1;
         end
%      end
end
end

Te=tfr1.*IF;
X=1
% %The following code is an alternative way to estimate IF.
% %In theroy, they are same.
% omega = zeros(round(N/2),tcol);
% for b=1:N
% omega(:,b) = (ft-1)'+real(va*1i*tfr2(ft,b)/2/pi./tfr1(ft,b));
% end
% for i=1:round(N/2)%frequency
% for j=1:N%time
%     if abs(tfr1(i,j))>0.8*E%default frequency resolution is 1Hz.
%         if abs(omega(i,j)-i)<0.5%default frequency resolution is 1Hz.
%         IF(i,j)=1;
%         end
%     end
% end
% end
