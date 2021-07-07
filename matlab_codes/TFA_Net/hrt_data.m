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
load target_bin_sig.mat
n_total_time=10000;
data_reshape=target_bin_sig(1:8:n_total_time).';
% data_reshape=data_reshape.*hamming(length(data_reshape));
t1=16*n_total_time/length(target_bin_sig);
t=0:t1/length(data_reshape):t1-t1/length(data_reshape);
fs=1/(16/n_total_time);
winlen=32;
ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;
x11=5.5; x22=8.5;
y11=-250;   y22=150;

%% STFT
window_v=ones(1,64);
spc_STFT=abs(spectrogram(data_reshape,winlen,winlen-1,nfft));
spc_STFT=sqrt(spc_STFT);
spc_STFT=spc_STFT/max(max(spc_STFT));
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
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_STFT),1)));

xlabel({'Time / sec';'(d)'})
ylabel('Freq. / Hz')
title(strcat('local STFT, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);

%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST=sqrt(abs(spc_SST));
spc_SST=spc_SST/max(max(spc_SST));
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
spc_SET=sqrt(abs(spc_SET));

spc_SET=spc_SET/max(max(spc_SET));
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

spc_MSST=sqrt(abs(spc_MSST1));
spc_MSST=spc_MSST/max(max(spc_MSST));
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
           ret=[ret;tmp(overlay/2:128-overlay/2,:)];
        elseif i==segment_num
           ret=[ret;tmp(end-(left_point-overlay/2-1):(end-overlay/2),:)];
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
ret=ret/max(max(ret));
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