
clc
clear
close all
nfft=256;
fsz=16;
clims_lim=40;


load dianji_data.mat
data_reshape=(dianji_data(1:2:8100).').';
sz=size(data_reshape);
fs=4e6;
dt=1/fs;
FS=fs;
t=(dt:dt:(sz)*dt)*1000;
winlen=32;
nfft=256;
corr_fs=1000;
ydelta=corr_fs/nfft;
yaxis=(0:ydelta:corr_fs-ydelta)-corr_fs/2;


x11=0.28; x22=0.43;
y11=-120;   y22=153;

x1=0.601;x2=1.0125;
y1=179;y2=485;
ylim_low=-200;
%% RS
spc_RS  = RS(data_reshape,winlen);
spc_RS=abs(spc_RS);

tt=t;
figure;
clims = [max(max(20*log10((abs(spc_RS)))))-40,max(max(20*log10((abs(spc_RS)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('RS, winlen=',num2str(winlen)))
xlabel({'Time / ms';'(d)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_RS_win',num2str(winlen));
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

figure;
clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
imagesc(t,yaxis,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('GWarblet'))
xlabel({'Time / ms';'(c)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);

imagesc(t,yaxis,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_GWarblet');
saveas(gcf, fname);

%% WVD
Spec=wvd(data_reshape,'smoothedPseudo');
Spec=fftshift(Spec,1);
figure;
clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
imagesc(t,yaxis/2,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('SPWVD'))
xlabel({'Time / ms';'(b)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);

imagesc(t,yaxis/2,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_SPWVD');
saveas(gcf, fname);

%% cwt
% Spec=cwt((data_reshape));
% Spec=([(fftshift(Spec(:,:,2),1));fftshift(flipud(Spec(:,:,1)),1)]);
% figure;
% clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
% imagesc(t,yaxis,20*log10((abs(Spec))));
% set(gca,'ydir','normal')
% ylim([ylim_low max(yaxis)])
% title(strcat('CWT'))
% xlabel({'Time / ms';'(c)'})
% ylabel('Freq. / Hz')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
% rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
% line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
% line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
% axes('Position',[0.6,0.6,0.3,0.3]);
% 
% imagesc(t,yaxis,20*log10((abs(Spec))));
% set(gca,'ydir','normal')
% set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_CWT');
% saveas(gcf, fname);
%% STFT
data_reshape1=[zeros(winlen/2,1);data_reshape;zeros(winlen/2-1,1)];
spc_STFT=abs(stft(data_reshape1,'Window',hamming(winlen).','OverlapLength',winlen-1,'FFTLength',nfft));
tt=t;

figure;
clims = [max(max(20*log10((abs(spc_STFT)))))-clims_lim,max(max(20*log10((abs(spc_STFT)))))];
imagesc(tt,yaxis,20*log10((abs(spc_STFT))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('STFT, winlen=',num2str(winlen)))
xlabel({'Time / ms';'(a)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);

imagesc(tt,yaxis,20*log10(abs(spc_STFT)),clims)
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_STFT_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_STFT_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);
%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST=abs(spc_SST);

tt=t;
figure;
clims = [max(max(20*log10((abs(spc_SST)))))-25,max(max(20*log10((abs(spc_SST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / ms';'(e)'})
ylabel('Freq. / Hz')
title(strcat('SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_SST_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_SST_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% SET
[spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
spc_SET=abs(spc_SET);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_SET)))))-15,max(max(20*log10((abs(spc_SET)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('SET, winlen=',num2str(winlen)))
xlabel({'Time / ms';'(f)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_SET_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_SET_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% MSST
[spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
spc_MSST=spc_MSST1;
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_MSST)))))-20,max(max(20*log10((abs(spc_MSST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('MSST, winlen=',num2str(winlen)))
xlabel({'Time / ms';'(g)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_MSST_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_MSST_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);



%% ResFreq
data_rsh=data_reshape.';
mv=max(abs(data_rsh));
noisedSig=data_rsh;

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
imagesc(t,yaxis,20*log10(fftshift(abs(ret.'),1)),clims)
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / ms';'(i)'})
ylabel('Freq. / Hz')
title('TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x1],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x1],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.6,0.6,0.3,0.3]);
imagesc(tt,yaxis,20*log10(fftshift(abs(ret.'),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp3_dianji_TFA-Net');
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_TFA-Net','.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);
%% 

