clc
clear
close all
N=128;
nfft=128;

% h=figure();
% set(h,'position',[100 100 1200 900]);
% ha=tight_subplot(4,2,[0.09 0.05],[.1 .05],[.05 .02]);
fsz=16;
%% SIMU1
fs=128;
ts=1/fs;
t = 0 : ts : 1-ts;

c1 = 2 * pi * 14;            % initial frequency of the chirp excitation
c2 = 2 * pi * 5/2;           % set the speed of frequency change be 1 Hz/second
c3 = 2 * pi * 1/3;
c4 = 2 * pi * -150/2;

d1 = -2 * pi * 14;            % initial frequency of the chirp excitation
d2 = 2 * pi * 5/2;           % set the speed of frequency change be 1 Hz/second
d3 = 2 * pi * 1/3;
d4 = 2 * pi * 150/2;

e1 = 2 * pi * 10;            % initial frequency of the chirp excitation
e2 = 2 * pi * 2;           % set the speed of frequency change be 1 Hz/second
e3 = 2 * pi * (-0.07);
e4 = 2 * pi * (-1.5);

Sig1 = exp(1i*(c1 * t + c2 * t.^2 / 2 + c3 * t.^3 /3 + c4 * t.^4 /4));   % get the A(t)
Sig2 = exp(1i*(d1 * t + d2 * t.^2 / 2 + d3 * t.^3 /3 + d4 * t.^4 /4));   % get the A(t)
Sig3 = exp(1i*(e1 * t + e2 * t.^2  + e3 * t.^3 + e4 *sin(6.5*pi*t) ));   % get the A(t)
Sig=Sig1+Sig2+Sig3;
Sig=awgn(Sig,20);
data_reshape=Sig.';
t1=t;

ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;
if1=(c1+c2.*t+c3 * t.^2+c4 * t.^3)/2/pi;
if2=(d1  + d2 * t  + d3 * t.^2 + d4 * t.^3)/2/pi;
if3=(e1  + 2*e2 * t  + 3*e3 * t.^2 + e4 *cos(6.5*pi*t)*6.5*pi)/2/pi ;
figure
plot(t,if1)
hold on
plot(t,if2)
plot(t,if3)
set(gca,'ydir','reverse')
ylim([min(yaxis) max(yaxis)])
xlabel({'Time / sec';'(a)'})
ylabel('Freq. / Hz')
title('Ground Truth')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
winlen=24;
x11=0.45; x22=0.65;
y11=-20;   y22=20;

%% STFT
window_v=ones(1,64);
spc_STFT=abs(spectrogram(data_reshape,winlen,winlen-1,nfft));
siglen2=size(spc_STFT,2);
siglen=size(data_reshape,1);
start_pos=fix((siglen-siglen2)/2);
if start_pos==0
    start_pos=1;
end
tt=t(start_pos:start_pos+siglen2-1);
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_STFT),1)));
title(strcat('STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(b)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST=abs(spc_SST);
siglen2=size(spc_SST,2);
start_pos=fix((siglen-siglen2)/2);
if start_pos==0
    start_pos=1;
end
tt=t(start_pos:start_pos+siglen2-1);
% spc_SST(spc_SST>0.001*max(max(spc_SST)))=0.001*max(max(spc_SST));
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
xlabel({'Time / sec';'(c)'})
ylabel('Freq. / Hz')
title(strcat('SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
xlabel({'Time / sec';'(d)'})
ylabel('Freq. / Hz')
title(strcat('local SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

%% SET
[spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
spc_SET=abs(spc_SET);
siglen2=size(spc_SET,2);
start_pos=fix((siglen-siglen2)/2);
if start_pos==0
    start_pos=1;
end
tt=t(start_pos:start_pos+siglen2-1);
% spc_SET(spc_SET>0.001*max(max(spc_SET)))=0.001*max(max(spc_SET));
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SET),1)));
title(strcat('SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(e)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SET),1)));
title(strcat('local SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(f)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
%% MSST
[spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
% spc_MSST1=abs(spc_MSST1);
% [spc_MSST2,tfr,omega2]  = MSST(conj(data_reshape),winlen,3);
% spc_MSST2=abs(spc_MSST2);
spc_MSST=spc_MSST1;
siglen2=size(spc_MSST,2);
start_pos=fix((siglen-siglen2)/2);
if start_pos==0
    start_pos=1;
end
tt=t(start_pos:start_pos+siglen2-1);
% spc_MSST(spc_MSST>0.001*max(max(spc_MSST)))=0.001*max(max(spc_MSST));
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_MSST),1)));
title(strcat('MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(g)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_MSST),1)));
title(strcat('local MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(h)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

%% ResFreq
data_rsh=data_reshape.';
mv=max(abs(data_rsh));
noisedSig=data_rsh./mv;
ret=[];
overlay=16;
segment_num=ceil(length(noisedSig)/(128-overlay));
left_point=(length(noisedSig)-(segment_num-1)*(128-overlay));
for i=1:segment_num
    if i==segment_num
        ret=[ret;noisedSig(end-127:end)];
    else
        ret=[ret;noisedSig((128-overlay)*(i-1)+1:(128-overlay)*(i-1)+1+127)];
    end
end
if segment_num==0
    bz=1;
    ret=noisedSig;
else
    bz=segment_num;
end

if ~exist('matlab_real2.h5','file')==0
    delete('matlab_real2.h5')
end
if ~exist('matlab_imag2.h5','file')==0
    delete('matlab_imag2.h5')   
end
if ~exist('bz.h5','file')==0
    delete('bz.h5')   
end
h5create('matlab_real2.h5','/matlab_real2',size(ret));
h5write('matlab_real2.h5','/matlab_real2',real(ret));
h5create('matlab_imag2.h5','/matlab_imag2',size(ret));
h5write('matlab_imag2.h5','/matlab_imag2',imag(ret));
h5create('bz.h5','/bz',size(bz));
h5write('bz.h5','/bz',bz);
% system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe resfreq_model.py')
flag=system('curl -s 127.0.0.1:5015/');
load data1_resfreq.mat
ret=[];

if segment_num==0
    tmp=squeeze(data1_resfreq(1,:,:));
    ret=[ret;tmp];
else
    for i=1:segment_num
        tmp=squeeze(data1_resfreq(i,:,:));
        if i==1
           ret=[ret;tmp(1:128-overlay/2,:)];
        elseif i==segment_num
           ret=[ret;tmp(end-(left_point-overlay/2-1):(end),:)];
        else
           ret=[ret;tmp(overlay/2:128-overlay/2,:)];
        end
    end
end
siglen2=size(ret,1);
start_pos=fix((siglen-siglen2)/2);
if start_pos==0
    start_pos=1;
end
tt=t(start_pos:start_pos+siglen2-1);
% ret(ret>0.01*max(max(ret)))=0.01*max(max(ret));
figure;
imagesc(tt,yaxis,(abs(ret.')))
xlabel({'Time / sec';'(i)'})
ylabel('Freq. / Hz')
title('TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
% figure;mesh(20*log10(abs((ret.'))))
figure;
imagesc(tt,yaxis,(abs(ret.')))
xlabel({'Time / sec';'(j)'})
ylabel('Freq. / Hz')
title('local TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);