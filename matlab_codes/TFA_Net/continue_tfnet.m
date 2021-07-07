clc
clear
close all
N=128;
nfft=128;

% h=figure();
% set(h,'position',[100 100 1200 900]);
% ha=tight_subplot(4,2,[0.09 0.05],[.1 .05],[.05 .02]);
fsz=13;
%% verify weak tgt
% fs = 100;
% SampFreq=fs;
% t = 0:1/fs:6-1/fs;
% Sig1 = 50*exp(1i*2*pi*6*t);   % get the A(t)
% Sig2 = exp(1i*2*pi*0*t);
% Sig=Sig1+Sig2;
% % Sig=awgn(Sig,0);
% data_reshape=Sig.';
% ydelta=fs/nfft;
% yaxis=(0:ydelta:fs-ydelta)-fs/2;
% if1=6*ones(1,length(t));
% if2=0*ones(1,length(t));
% 
% figure;
% plot(t,if1);
% hold on;
% plot(t,if2);
% xlabel('(a)')
% ylabel('Doppler / Hz')
% title('Ground Truth')
% set(gca,'ydir','reverse')
% ylim([min(yaxis) max(yaxis)])
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% winlen=64;

%% verify the winlen
% fs = 100;
% SampFreq=fs;
% t = 0:1/fs:6-1/fs;
% Sig = exp(1i*2*pi*(8 * t +6 * sin(0.9*t)))+exp(1i*2*pi*(10*t+6*sin(1.2*t)));   % get the A(t)
% Sig=[Sig];
% 
% data_reshape=Sig.';
% 
% ydelta=fs/nfft;
% yaxis=(0:ydelta:fs-ydelta)-fs/2;
% if1=8+6*0.9*cos(0.9*t);
% if2=10+6*1.2*cos(1.2*t);
% 
% figure;
% plot(t,if1);
% hold on;
% plot(t,if2);
% xlabel('(a)')
% ylabel('Doppler / Hz')
% title('Ground Truth')
% set(gca,'ydir','reverse')
% ylim([min(yaxis) max(yaxis)])
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% winlen=32;
%% test single component
fs = 250;
SampFreq=fs;
t = 0:1/fs:3-1/fs;
c1 = -2 * pi * 10;            % initial frequency of the chirp excitation
c2 = 2 * pi * 5;           % set the speed of frequency change be 1 Hz/second
c3 = 2 * pi * (-0.07);
c4 = 2 * pi * 2;
tt1=t(1:fs/2);
tt2=t(fs/2+1:end);
Sig = exp(1i*(c1 * t + c2 * t.^2  + c3 * t.^3 + c4 * sin(1*pi*t)));   % get the A(t)
Sig=[Sig];
% Sig=awgn(Sig,60);
data_reshape=Sig.';

ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;
if1=(c1  + 2*c2 * t  + 3*c3 * t.^2 + c4 * cos(5*pi*t)*5*pi)/2/pi;
x11=0.728; x22=0.876;
y11=10;   y22=30;

figure;
plot(t,if1)
xlabel('(a)')
ylabel('Doppler / Hz')
title('Ground Truth')
set(gca,'ydir','reverse')
ylim([min(yaxis) max(yaxis)])
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
winlen=8  ;
%% SIMU1
% fs = 128;
% SampFreq=fs;
% t = 0:1/fs:1-1/fs;
% c1 = 2 * pi * 10;            % initial frequency of the chirp excitation
% c2 = 2 * pi * 2;           % set the speed of frequency change be 1 Hz/second
% c3 = 2 * pi * (-0.07);
% c4 = 2 * pi * 1;
% 
% d1 = 2 * pi * 10;            % initial frequency of the chirp excitation
% d2 = 2 * pi * 2;           % set the speed of frequency change be 1 Hz/second
% d3 = 2 * pi * (-0.07);
% d4 = 2 * pi * (-1);
% Sig1 = 1*exp(1i*(c1 * t + c2 * t.^2  + c3 * t.^3 + c4 * sin(6.5*pi*t)));   % get the A(t)
% Sig2 = 1*exp(1i*(d1 * t + d2 * t.^2  + d3 * t.^3 + d4 *sin(6.5*pi*t) ));   % get the A(t)
% Sig=Sig1+Sig2;
% % Sig=Sig.*hamming(length(Sig)).';
% data_reshape=Sig.';
% t1=1;
% 
% ydelta=fs/nfft;
% yaxis=(0:ydelta:fs-ydelta)-fs/2;
% if1=(c1  + 2*c2 * t  + 3*c3 * t.^2 + c4 * cos(6.5*pi*t)*6.5*pi)/2/pi;
% if2=(d1  + 2*d2 * t.^1  +3*d3 * t.^2 + d4 *cos(6.5*pi*t)*6.5*pi)/2/pi;
% 
% figure;
% plot(t,if1)
% hold on
% plot(t,if2)
% set(gca,'ydir','reverse')
% ylim([min(yaxis) max(yaxis)])
% xlabel({'Time / sec';'(a)'})
% ylabel('Doppler / Hz')
% title('Ground Truth')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% winlen=16;
%% SIMU2
% fs=128;
% ts=1/fs;
% t = 0 : ts : 1-ts;
% 
% c1 = 2 * pi * 14;            % initial frequency of the chirp excitation
% c2 = 2 * pi * 5/2;           % set the speed of frequency change be 1 Hz/second
% c3 = 2 * pi * 1/3;
% c4 = 2 * pi * -150/2;
% 
% d1 = -2 * pi * 14;            % initial frequency of the chirp excitation
% d2 = 2 * pi * 5/2;           % set the speed of frequency change be 1 Hz/second
% d3 = 2 * pi * 1/3;
% d4 = 2 * pi * 150/2;
% 
% e1 = 2 * pi * 10;            % initial frequency of the chirp excitation
% e2 = 2 * pi * 2;           % set the speed of frequency change be 1 Hz/second
% e3 = 2 * pi * (-0.07);
% e4 = 2 * pi * (-1.5);
% 
% Sig1 = exp(1i*(c1 * t + c2 * t.^2 / 2 + c3 * t.^3 /3 + c4 * t.^4 /4));   % get the A(t)
% Sig2 = exp(1i*(d1 * t + d2 * t.^2 / 2 + d3 * t.^3 /3 + d4 * t.^4 /4));   % get the A(t)
% Sig3 = exp(1i*(e1 * t + e2 * t.^2  + e3 * t.^3 + e4 *sin(6.5*pi*t) ));   % get the A(t)
% Sig=Sig1+Sig2+Sig3;
% % Sig=Sig.*hamming(length(Sig)).';
% data_reshape=Sig.';
% t1=t;
% 
% ydelta=fs/nfft;
% yaxis=(0:ydelta:fs-ydelta)-fs/2;
% if1=(c1+c2.*t+c3 * t.^2+c4 * t.^3)/2/pi;
% if2=(d1  + d2 * t  + d3 * t.^2 + d4 * t.^3)/2/pi;
% if3=(e1  + 2*e2 * t  + 3*e3 * t.^2 + e4 *cos(6.5*pi*t)*6.5*pi)/2/pi ;
% figure
% plot(t,if1)
% hold on
% plot(t,if2)
% plot(t,if3)
% set(gca,'ydir','reverse')
% ylim([min(yaxis) max(yaxis)])
% xlabel({'Time / sec';'(a)'})
% ylabel('Doppler / Hz')
% title('Ground Truth')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% winlen=16;
%% ball
% load ball_0426_chan1.mat
% sig=chan1(3,:);
% n_total_time=10000;
% fs=1/(16/n_total_time);
% data_reshape=sig(1:2:n_total_time).';
% t1=16*n_total_time/length(sig);
% t=0:t1/length(data_reshape):t1-t1/length(data_reshape);
% winlen=32;
% ydelta=fs/nfft;
% yaxis=(0:ydelta:fs-ydelta)-fs/2;
%% people 3 breathe only
% load target_bin_sig.mat
% n_total_time=10000;
% data_reshape=target_bin_sig(1:8:n_total_time).';
% % data_reshape=data_reshape.*hamming(length(data_reshape));
% t1=16*n_total_time/length(target_bin_sig);
% t=0:t1/length(data_reshape):t1-t1/length(data_reshape);
% fs=1/(16/n_total_time);
% winlen=32;
% ydelta=fs/nfft;
% yaxis=(0:ydelta:fs-ydelta)-fs/2;

%% people 1
load bre_hrt_data2.mat
n_total_time=10000;
data_reshape=bre_hrt_data2;
data_reshape=data_reshape.';
% data_reshape=data_reshape.*hamming(length(data_reshape));
sz=size(data_reshape);
time=8;
dt=time/50000;
t1=128*390*dt;
t=0:t1/length(data_reshape):t1-t1/length(data_reshape);
fs=1/(time/n_total_time);
winlen=16;
ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;
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
title('STFT')
xlabel({'Time / sec';'(b)'})
ylabel('Doppler / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% figure;mesh(20*log10(fftshift(abs(spc_STFT),1)))
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
ylabel('Doppler / Hz')
title('SST')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',1);
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
xlabel({'Time / sec';'(c)'})
ylabel('Doppler / Hz')
title('SST')
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
title('SET')
xlabel({'Time / sec';'(d)'})
ylabel('Doppler / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
%% LMSST
% [spc_LMSST,IF,TFR]  = LMSST_Y2(data_reshape,winlen);
% spc_LMSST=abs(spc_LMSST);
% siglen2=size(spc_LMSST,2);
% start_pos=fix((siglen-siglen2)/2);
% if start_pos==0
%     start_pos=1;
% end
% tt=t(start_pos:start_pos+siglen2-1);
% figure;
% % spc_LMSST(spc_LMSST>0.001*max(max(spc_LMSST)))=0.001*max(max(spc_LMSST));
% imagesc(tt,yaxis,(fftshift(abs(spc_LMSST),1)));
% xlabel({'Time / sec';'(e)'})
% ylabel('Doppler / Hz')
% title('LMSST')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
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
title('MSST')
xlabel({'Time / sec';'(f)'})
ylabel('Doppler / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
%% RS
% spc_RS  = RS(data_reshape,winlen);
% spc_RS=abs(spc_RS);
% siglen2=size(spc_RS,2);
% start_pos=fix((siglen-siglen2)/2);
% if start_pos==0
%     start_pos=1;
% end
% tt=t(start_pos:start_pos+siglen2-1);
% figure;
% imagesc(tt,yaxis,(fftshift(abs(spc_RS),1)));
% xlabel({'Time / sec';'(g)'})
% ylabel('Doppler / Hz')
% title('RS')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
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
           ret=[ret;tmp(1:128-overlay/2,:)];
        elseif i==segment_num
           ret=[ret;tmp(end-(left_point-overlay/2):(end-overlay),:)];
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
xlabel({'Time / sec';'(b)'})
ylabel('Doppler / Hz')
title('TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
% figure;mesh(20*log10(abs((ret.'))))