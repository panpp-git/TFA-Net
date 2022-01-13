function [Ts,tfr,omega2] = MSST_Y_new2(x,hlength,num);
% Computes the MSST (Ts)  of the signal x.
% Expression (31)-based Algorithm.
% INPUT
%    x      :  Signal needed to be column vector.
%    hlength:  The hlength of window function.
%    num    :  iteration number.
% OUTPUT
%    Ts     :  The SST
%    tfr     :  The STFT
nfft=256;
[xrow,xcol] = size(x);

if (xcol~=1),
    error('X must be column vector');
end;

if (nargin < 3),
    error('At least 3 parameter is required');
end
%
% if (nargin < 2),
% hlength=round(xrow/5);
% num=1;
% else if (nargin < 3),
% num=1;
% end;

hlength=hlength+1-rem(hlength,2);
ht = linspace(-0.5,0.5,hlength);ht=ht';

% Gaussian window
% h = exp(-pi/0.32^2*ht.^2);
h = kaiser(min(hlength,xrow),10);
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

[neta,nb]=size(tfr1);

if num>1
    for kk=1:num-1
        for b=1:nb
            for eta=1:neta
                k = omega(eta,b);
                if k>=1 && k<=neta
                    omega2(eta,b)=omega(k,b);
                end
            end
        end
        omega=omega2;
    end
else
    omega2=omega;
end

for b=1:nb%time
    % Reassignment step
    for eta=1:neta%frequency
        if abs(sstout(eta,b))>0.0001%you can set much lower value than this.
            k = omega2(eta,b);
            if k>=1 && k<=neta
                Ts(k,b) = Ts(k,b) + sstout(eta,b);
            end
        end
    end
end
%tfr=tfr/(sum(h)/2);
tfr=sstout/(xrow/2);
Ts=Ts/(xrow/2);
end
%
% function [Ts_f]=SST(tfr_f,omega_f);
% [tfrm,tfrn]=size(tfr_f);
% Ts_f= zeros (tfrm,tfrn) ;
% %mx=max(max(tfr_f));
% for b=1:tfrn%time
%     % Reassignment step
%     for eta=1:tfrm%frequency
%         %if abs(tfr_f(eta,b))>0.001*mx%you can set much lower value than this.
%             k = omega_f(eta,b);
%             if k>=1 && k<=tfrm
%                 Ts_f(k,b) = Ts_f(k,b) + tfr_f(eta,b);
%             end
%         %end
%     end
% end
% end