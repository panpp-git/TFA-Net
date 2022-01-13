
clc
clear
close all
nfft=256;
fsz=16;
clims_lim=50;


load 1110_zzq_80_1_Raw_0_chan1.mat
data_reshape=sig.';
sz=size(data_reshape);
time=16.25;
dt=time/sz(1);
t=0:dt:time-dt;
fs=1/dt;
FS=fs;
TS=dt;
winlen=32;
ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;


x11=9; x22=11.7;
y11=-18;   y22=18;

x1=0.55;x2=7;
y1=11.8;y2=37.4;
ylim_low=-19;

%% RS
spc_RS  = RS(data_reshape,winlen);
spc_RS=abs(spc_RS);
spc_RS=flipud(spc_RS);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_RS)))))-70,max(max(20*log10((abs(spc_RS)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('RS, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(d)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_RS_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_MSST_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);


%% GWarblet
[Spec,f] = GWarblet_complex(data_reshape,FS,0,1,nfft,winlen);
[v l] = max(Spec,[],1);
[IF, a_n,b_n,fm] = get_fscoeff(f(l),length(t),t,FS);
WinLen =winlen*4;
[Spec,f] = GWarblet_complex(data_reshape,FS,[-a_n;b_n],fm(2:end),nfft,WinLen);

[Spec2,f] = GWarblet_complex(conj(data_reshape),FS,0,1,nfft,winlen);
[v l] = max(Spec2,[],1);
[IF, a_n,b_n,fm] = get_fscoeff(f(l),length(t),t,FS);
WinLen =winlen*4;
[Spec2,f] = GWarblet_complex(conj(data_reshape),FS,[-a_n;b_n],fm(2:end),nfft,WinLen);
Spec=[flipud(Spec2);Spec];
Spec=flipud(Spec);
figure;
clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
imagesc(t,yaxis,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('GWarblet'))
xlabel({'Time / sec';'(c)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);

imagesc(t,yaxis,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_GWarblet');
saveas(gcf, fname);

%% WVD
Spec=wvd(data_reshape,'smoothedPseudo');
Spec=fftshift(Spec,1);
Spec=flipud(Spec);
figure;
clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
imagesc(t,yaxis/2,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('SPWVD'))
xlabel({'Time / sec';'(b)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);

imagesc(t,yaxis/2,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_SPWVD');
saveas(gcf, fname);

%% cwt
% Spec=cwt((data_reshape));
% Spec=([(fftshift(Spec(:,:,2),1));fftshift(flipud(Spec(:,:,1)),1)]);
% Spec=flipud(Spec);
% figure;
% clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
% imagesc(t,yaxis,20*log10((abs(Spec))),clims);
% set(gca,'ydir','normal')
% ylim([ylim_low max(yaxis)])
% title(strcat('CWT'))
% xlabel({'Time / sec';'(c)'})
% ylabel('Freq. / Hz')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
% rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
% line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
% line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
% axes('Position',[0.16,0.6,0.3,0.3]);
% 
% imagesc(t,yaxis,20*log10((abs(Spec))),clims);
% set(gca,'ydir','normal')
% set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_CWT');
% saveas(gcf, fname);
%% STFT
data_reshape1=[zeros(winlen/2,1);data_reshape;zeros(winlen/2-1,1)];
spc_STFT=abs(stft(data_reshape1,'Window',hamming(winlen).','OverlapLength',winlen-1,'FFTLength',nfft));
spc_STFT=flipud(spc_STFT);
tt=t;

figure;
clims = [max(max(20*log10((abs(spc_STFT)))))-clims_lim,max(max(20*log10((abs(spc_STFT)))))];
imagesc(tt,yaxis,20*log10((abs(spc_STFT))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(a)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);

imagesc(tt,yaxis,20*log10(abs(spc_STFT)),clims)
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_STFT_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_STFT_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);
%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST=abs(spc_SST);
spc_SST=flipud(spc_SST);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_SST)))))-70,max(max(20*log10((abs(spc_SST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / sec';'(e)'})
ylabel('Freq. / Hz')
title(strcat('SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_SST_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_SST_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% SET
[spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
spc_SET=abs(spc_SET);
spc_SET=flipud(spc_SET);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_SET)))))-clims_lim,max(max(20*log10((abs(spc_SET)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(f)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_SET_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_SET_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% MSST
[spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
spc_MSST=spc_MSST1;
spc_MSST=flipud(spc_MSST);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_MSST)))))-clims_lim,max(max(20*log10((abs(spc_MSST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(g)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_MSST_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_MSST_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);



%% ResFreq
data_rsh=data_reshape.';
mv=max(abs(data_rsh));
noisedSig=data_rsh/mv;

siglen=256;
ret=[];
sp=1;
ep=sp+siglen-1;
overlay=30;
while ep<length(noisedSig)
    ret=[ret;noisedSig(sp:ep)];
    sp=ep+1-overlay;
    ep=sp+siglen-1;
end
flag=0;
if ep>length(noisedSig) && length(noisedSig)-sp+1>overlay/2
    flag=1;
    left_len=length(noisedSig)-sp+1;
    ret=[ret;noisedSig(end-siglen+1:end)];
end

bz=size(ret,1);

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

net_flag=system('curl -s 127.0.0.1:5012/');
load data1_resfreq.mat
ret=[];

if flag==1
    iter=bz-1;
else
    iter=bz;
end
for i=1:iter
    if i==1
        tmp=squeeze(data1_resfreq(i,:,:));
        ret=[ret;tmp(1:siglen-overlay/2,:)];
    else
        tmp=squeeze(data1_resfreq(i,:,:));
        ret=[ret;tmp(overlay/2+1:siglen-overlay/2,:)];
    end
end
if flag==1
    tmp=squeeze(data1_resfreq(iter+1,:,:));
    tmp=tmp(end-left_len+1:end,:);
    ret=[ret;tmp(overlay/2+1:end,:)];
end
figure;
ret=(10.^(ret/20)-1)/10;

clims = [max(max(20*log10((abs(ret)))))-clims_lim,max(max(20*log10((abs(ret)))))];
imagesc(t,yaxis,flipud(20*log10(fftshift(abs(ret.'),1))),clims)
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / sec';'(i)'})
ylabel('Freq. / Hz')
title('TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.16,0.6,0.3,0.3]);
imagesc(tt,yaxis,flipud(20*log10(fftshift(abs(ret.'),1))),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_breheart_TFA-Net');
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_TFA-Net','.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);
