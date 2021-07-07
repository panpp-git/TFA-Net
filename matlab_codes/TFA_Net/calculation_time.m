clc
clear
close all
N=128;
nfft=128;

% h=figure();
% set(h,'position',[100 100 1200 900]);
% ha=tight_subplot(4,2,[0.09 0.05],[.1 .05],[.05 .02]);
fsz=16;
%% test single component
fs = 250;
SampFreq=fs;
t = 0:1/fs:10-1/fs;
c1 = -2 * pi * 10;            % initial frequency of the chirp excitation
c2 = 2 * pi * 5;           % set the speed of frequency change be 1 Hz/second
c3 = 2 * pi * (-0.07);
c4 = 2 * pi * 2;
tt1=t(1:fs/2);
tt2=t(fs/2+1:end);
Sig = exp(1i*(c1 * t + c2 * t.^2  + c3 * t.^3 + c4 * sin(5*pi*t)));   % get the A(t)
Sig=[Sig];
% Sig=awgn(Sig,60);
data_reshape=Sig.';

ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;
winlen=8;
%% STFT


tic
for i=10000
    spc_STFT=abs(spectrogram(data_reshape,winlen,winlen-1,nfft));
end
b=toc;
STFT_time=b;

%% SST
tic
for i=10000
    spc_SST  = SST2(data_reshape,winlen);
    spc_SST=abs(spc_SST);
end
b=toc;
SST_time=b;

%% SET

tic
for i=10000
    [spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
    spc_SET=abs(spc_SET);
end
b=toc;
SET_time=b;

%% MSST

tic
for i=10000
    [spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
    spc_MSST=abs(spc_MSST1);
end
b=toc;
MSST_time=b;

%% ResFreq
tic
for i=10000
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
    flag=system('curl -s 127.0.0.1:5010/');
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
               ret=[ret;tmp(end-(left_point-overlay/2):(end),:)];
            else
               ret=[ret;tmp(overlay/2:128-overlay/2,:)];
            end
        end
    end
    ret=abs(ret);
end
b=toc;
TFA_time=b;