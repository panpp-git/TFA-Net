clc
clear
close all
N=256;
nfft=256;

% h=figure();
% set(h,'position',[100 100 1200 900]);
% ha=tight_subplot(4,2,[0.09 0.05],[.1 .05],[.05 .02]);
fsz=23;
%% SIMU1
fs=100;
ts=1/fs;
t = 0 : ts : 10-ts;
Sig1 = exp(1i*2*pi*(8* t + 6 *sin(t) ));   % get the A(t)
Sig2 = exp(1i*2*pi*(10 * t + 6 *sin(1.5*t) ));   % get the A(t)
Sig=Sig1+Sig2;
data_reshape=Sig.';
t1=t;

ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;

if1=(8 + 6*1 *cos(1*t)) ;
if2=(10 + 6*1.5 *cos(1.5*t)) ;

h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,2,[0.08 0.07],[.31 .1],[.07 .02]);
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
x11=4.4; x22=8.5;
y11=6;   y22=15;

xx11=0.5; xx22=3;
yy11=0.5;   yy22=17;
ylow=0;
yhigh=20;
ylim([ylow,yhigh])

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
ylim([ylow,yhigh])

fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','ref_STFT_win',num2str(winlen));
saveas(gcf, fname);

%% GT
xx=(fftshift(abs(spc_STFT),1));
x1=(xx(:,371));
x2=(xx(:,211));
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,2,[0.08 0.07],[.31 .1],[.06 .02]);
axes(ha(1))
axes(ha(1))
plot(yaxis,x2,'k','Linewidth',2);
hold on
plot([if1(241),if1(241)],[0,55],'r','Linewidth',2)
plot([if2(241),if2(241)],[0,55],'r','Linewidth',2)
xlabel({'Freq. / Hz';'(a)'})
ylabel('Amp.')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
xlim([-5,15])
ylim([ylow,55])
grid on
axes(ha(2))
plot(yaxis,x1,'k','Linewidth',2);
hold on
plot([if1(401),if1(401)],[0,50],'r','Linewidth',2)
plot([if2(401),if2(401)],[0,50],'r','Linewidth',2)
xlabel({'Freq. / Hz';'(b)'})
ylabel('Amp.')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
xlim([0,25])
ylim([ylow,50])
grid on

fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','ref_GT_win',num2str(winlen));
saveas(gcf, fname);
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
ha=tight_subplot(1,3,[0.05 0.05],[.31 .1],[.05 .02]);
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
ylim([ylow,yhigh])

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

fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','ref_SST_win',num2str(winlen));
saveas(gcf, fname);
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
ha=tight_subplot(1,3,[0.05 0.05],[.31 .1],[.05 .02]);
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
ylim([ylow,yhigh])

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

fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','ref_SET_win',num2str(winlen));
saveas(gcf, fname);
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
ha=tight_subplot(1,3,[0.05 0.05],[.31 .1],[.05 .02]);
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
ylim([ylow,yhigh])

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

fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','ref_MSST_win',num2str(winlen));
saveas(gcf, fname);

