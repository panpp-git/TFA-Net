clc
clear
close all
N=128;
nfft=1600;

% h=figure();
% set(h,'position',[100 100 1200 900]);
% ha=tight_subplot(4,2,[0.09 0.05],[.1 .05],[.05 .02]);
fsz=18;
%% SIMU1
fs=100;
ts=1/fs;
t = 0 : ts : 10-ts;
Sig1 = exp(1i*2*pi*(8* t + 6 *sin(0.9*t) ));   % get the A(t)
Sig2 = exp(1i*2*pi*(10 * t + 6 *sin(1.3*t) ));   % get the A(t)
Sig=Sig1+Sig2;
data_reshape=Sig.';
t1=t;

ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;

if1=(8 + 6*0.9 *cos(0.9*t)) ;
if2=(10 + 6*1.3 *cos(1.3*t)) ;

h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,2,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1))
plot(t,if1)
hold on
plot(t,if2)

set(gca,'ydir','reverse')
ylim([min(yaxis) max(yaxis)])
xlabel({'Time / sec';'(a)'})
ylabel('Freq. / Hz')
title('Ground Truth')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
winlen=64;
x11=5; x22=6.8;
y11=7;   y22=17;

xx11=0.5; xx22=4;
yy11=1;   yy22=19;

ylim([0,20])

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

axes(ha(2))
imagesc(tt,yaxis,(fftshift(abs(spc_STFT),1)));
title(strcat('STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(b)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
ylim([0,20])

FSZ=24;
xx=(fftshift(abs(spc_STFT),1));
x1=(xx(:,371));
x2=(xx(:,211));
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,2,[0.09 0.08],[.25 .06],[.07 .02]);
axes(ha(1))
axes(ha(1))
plot(yaxis,x2,'k','Linewidth',2);
hold on
plot([if1(241),if1(241)],[0,50],'r','Linewidth',2)
plot([if2(241),if2(241)],[0,50],'r','Linewidth',2)
xlabel({'Freq. / Hz';'(a)'})
ylabel('Amp.')
set(gca,'FontSize',FSZ); 
set(get(gca,'XLabel'),'FontSize',FSZ);
set(get(gca,'YLabel'),'FontSize',FSZ);
xlim([-5,15])
ylim([0,50])
grid on
axes(ha(2))
plot(yaxis,x1,'k','Linewidth',2);
hold on
plot([if1(401),if1(401)],[0,50],'r','Linewidth',2)
plot([if2(401),if2(401)],[0,50],'r','Linewidth',2)
xlabel({'Freq. / Hz';'(b)'})
ylabel('Amp.')
set(gca,'FontSize',FSZ); 
set(get(gca,'XLabel'),'FontSize',FSZ);
set(get(gca,'YLabel'),'FontSize',FSZ);
xlim([0,20])
ylim([0,50])
grid on
%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST=abs(spc_SST);
siglen2=size(spc_SST,2);
start_pos=fix((siglen-siglen2)/2);
if start_pos==0
    start_pos=1;
end
tt=t(start_pos:start_pos+siglen2-1);

h=figure();
set(h,'position',[100 100 1600 400]);
ha=tight_subplot(1,3,[0.06 0.04],[.26 .08],[.04 .02]);
axes(ha(1))
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
xlabel({'Time / sec';'(c)'})
ylabel('Freq. / Hz')
title(strcat('SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[xx11 yy11 xx22-xx11 yy22-yy11],'EdgeColor','#D2691E','Linewidth',4);
ylim([0,20])

axes(ha(3))
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
xlabel({'Time / sec';'(e)'})
ylabel('Freq. / Hz')
title(strcat('local SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);

axes(ha(2))
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
xlabel({'Time / sec';'(d)'})
ylabel('Freq. / Hz')
title(strcat('local SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[xx11 xx22],'ylim',[yy11 yy22]);
rectangle('Position',[xx11 yy11 xx22-xx11 yy22-yy11],'EdgeColor','#D2691E','Linewidth',6);
%% SET
[spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
spc_SET=abs(spc_SET);
siglen2=size(spc_SET,2);
start_pos=fix((siglen-siglen2)/2);
if start_pos==0
    start_pos=1;
end
tt=t(start_pos:start_pos+siglen2-1);
h=figure();
set(h,'position',[100 100 1600 400]);
ha=tight_subplot(1,3,[0.06 0.04],[.26 .08],[.04 .02]);
axes(ha(1))
imagesc(tt,yaxis,(fftshift(abs(spc_SET),1)));
title(strcat('SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(f)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[xx11 yy11 xx22-xx11 yy22-yy11],'EdgeColor','#D2691E','Linewidth',4);
ylim([0,20])

axes(ha(3))
imagesc(tt,yaxis,(fftshift(abs(spc_SET),1)));
title(strcat('local SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(h)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',6);

axes(ha(2))
imagesc(tt,yaxis,(fftshift(abs(spc_SET),1)));
title(strcat('local SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(g)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[xx11 xx22],'ylim',[yy11 yy22]);
rectangle('Position',[xx11 yy11 xx22-xx11 yy22-yy11],'EdgeColor','#D2691E','Linewidth',7);
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

h=figure();
set(h,'position',[100 100 1600 400]);
ha=tight_subplot(1,3,[0.06 0.04],[.26 .08],[.04 .02]);
axes(ha(1))
imagesc(tt,yaxis,(fftshift(abs(spc_MSST),1)));
title(strcat('MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(i)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[xx11 yy11 xx22-xx11 yy22-yy11],'EdgeColor','#D2691E','Linewidth',4);
ylim([0,20])

axes(ha(3))
imagesc(tt,yaxis,(fftshift(abs(spc_MSST),1)));
title(strcat('local MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(k)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',6);

axes(ha(2))
imagesc(tt,yaxis,(fftshift(abs(spc_MSST),1)));
title(strcat('local MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(j)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
set(gca,'xlim',[xx11 xx22],'ylim',[yy11 yy22]);
rectangle('Position',[xx11 yy11 xx22-xx11 yy22-yy11],'EdgeColor','#D2691E','Linewidth',7);
%% ResFreq
% data_rsh=data_reshape.';
% mv=max(abs(data_rsh));
% noisedSig=data_rsh./mv;
% ret=[];
% overlay=16;
% segment_num=ceil(length(noisedSig)/(128-overlay));
% left_point=(length(noisedSig)-(segment_num-1)*(128-overlay));
% for i=1:segment_num
%     if i==segment_num
%         ret=[ret;noisedSig(end-127:end)];
%     else
%         ret=[ret;noisedSig((128-overlay)*(i-1)+1:(128-overlay)*(i-1)+1+127)];
%     end
% end
% if segment_num==0
%     bz=1;
%     ret=noisedSig;
% else
%     bz=segment_num;
% end
% 
% if ~exist('matlab_real2.h5','file')==0
%     delete('matlab_real2.h5')
% end
% if ~exist('matlab_imag2.h5','file')==0
%     delete('matlab_imag2.h5')   
% end
% if ~exist('bz.h5','file')==0
%     delete('bz.h5')   
% end
% h5create('matlab_real2.h5','/matlab_real2',size(ret));
% h5write('matlab_real2.h5','/matlab_real2',real(ret));
% h5create('matlab_imag2.h5','/matlab_imag2',size(ret));
% h5write('matlab_imag2.h5','/matlab_imag2',imag(ret));
% h5create('bz.h5','/bz',size(bz));
% h5write('bz.h5','/bz',bz);
% % system('D:\ProgramData\Anaconda3\envs\complexPytorch-gpu\python.exe resfreq_model.py')
% flag=system('curl -s 127.0.0.1:5010/');
% load data1_resfreq.mat
% ret=[];
% 
% if segment_num==0
%     tmp=squeeze(data1_resfreq(1,:,:));
%     ret=[ret;tmp];
% else
%     for i=1:segment_num
%         tmp=squeeze(data1_resfreq(i,:,:));
%         if i==1
%            ret=[ret;tmp(overlay/2:128-overlay/2,:)];
%         elseif i==segment_num
%            ret=[ret;tmp(end-(left_point-overlay/2-1):(end),:)];
%         else
%            ret=[ret;tmp(overlay/2:128-overlay/2,:)];
%         end
%     end
% end
% siglen2=size(ret,1);
% start_pos=fix((siglen-siglen2)/2);
% if start_pos==0
%     start_pos=1;
% end
% tt=t(start_pos:start_pos+siglen2-1);
% % ret(ret>0.01*max(max(ret)))=0.01*max(max(ret));
% figure;
% imagesc(tt,yaxis,(abs(ret.')))
% ylim([0,20])
% xlabel({'Time / sec';'(i)'})
% ylabel('Freq. / Hz')
% title('TFA-Net')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
% % figure;mesh(20*log10(abs((ret.'))))
% figure;
% imagesc(tt,yaxis,(abs(ret.')))
% xlabel({'Time / sec';'(j)'})
% ylabel('Freq. / Hz')
% title('local TFA-Net')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
