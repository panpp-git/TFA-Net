
clc
clear
close all
nfft=256;
fsz=16;
clims_lim=30;


[Y, FS]=audioread('ringtoneFrasers.mp3');
TS=1/FS;
N=length(Y(:,1));
data_reshape=Y(floor(N*1.4/10):1:floor(2.4*N/6),1);

data_reshape=hilbert(data_reshape);
M=length(data_reshape);
t=0:TS:M*TS-TS;
winlen=64;
ydelta=FS/nfft;
yaxis=((0:ydelta:FS-ydelta)-FS/2)/1000;
x11=0.2; x22=0.55;
y11=5;   y22=19;

x1=0.7;x2=1;
y1=5;y2=15;
ylim_low=0;



%% STFT
data_reshape1=[zeros(winlen/2,1);data_reshape;zeros(winlen/2-1,1)];
spc_STFT=abs(stft(data_reshape1,'Window',hamming(winlen).','OverlapLength',winlen-1,'FFTLength',nfft));
tt=t;

h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1));
clims = [max(max(20*log10((abs(spc_STFT)))))-clims_lim,max(max(20*log10((abs(spc_STFT)))))];
imagesc(tt,yaxis,20*log10((abs(spc_STFT))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(a)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2))
imagesc(tt,yaxis,20*log10(abs(spc_STFT)),clims)
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
title(strcat('local STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(b)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

axes(ha(3))
imagesc(tt,yaxis,20*log10(abs(spc_STFT)),clims)
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
title(strcat('local STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(c)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_STFT_win',num2str(winlen));
saveas(gcf, fname);


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
% save gwarblet.mat Spec
% load gwarblet.mat
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1))
clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
imagesc(t,yaxis,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('GWarblet'))
xlabel({'Time / sec';'(g)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2))
imagesc(t,yaxis,20*log10((abs(Spec))),clims);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
title(strcat('local GWarblet'))
xlabel({'Time / sec';'(h)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

axes(ha(3))
imagesc(t,yaxis,20*log10((abs(Spec))),clims);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
title(strcat('local GWarblet'))
xlabel({'Time / sec';'(i)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_GWarblet_win',num2str(winlen));
saveas(gcf, fname);

%% WVD
yaxis1=((0:ydelta:FS/2-ydelta))/1000;
Spec=wvd(data_reshape,'smoothedPseudo',hamming(winlen-1),'NumFrequencyPoints',nfft);

h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1))
clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
imagesc(t,yaxis1,20*log10((abs(Spec))),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis1)])
title(strcat('SPWVD'))
xlabel({'Time / sec';'(d)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2))
imagesc(t,yaxis1,20*log10((abs(Spec))),clims);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
title(strcat('local SPWVD'))
xlabel({'Time / sec';'(e)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

axes(ha(3))
imagesc(t,yaxis1,20*log10((abs(Spec))),clims);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
title(strcat('local SPWVD'))
xlabel({'Time / sec';'(f)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_PSWVD_win',num2str(winlen));
saveas(gcf, fname);

%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST=abs(spc_SST);

tt=t;
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1));
clims = [max(max(20*log10((abs(spc_SST)))))-clims_lim,max(max(20*log10((abs(spc_SST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / sec';'(j)'})
ylabel('Freq. / kHz')
title(strcat('SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2));
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
title(strcat('local SST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(k)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

axes(ha(3))
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
title(strcat('local SST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(m)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_SST_win',num2str(winlen));
saveas(gcf, fname);

%% SET
[spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
spc_SET=abs(spc_SET);
tt=t;
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1));
clims = [max(max(20*log10((abs(spc_SET)))))-clims_lim,max(max(20*log10((abs(spc_SET)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(n)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2))
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
title(strcat('local SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(o)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

axes(ha(3))
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
title(strcat('local SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(p)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_SET_win',num2str(winlen));
saveas(gcf, fname);

%% MSST
[spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
spc_MSST=spc_MSST1;
tt=t;
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1));
clims = [max(max(20*log10((abs(spc_MSST)))))-clims_lim,max(max(20*log10((abs(spc_MSST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(q)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2))
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
title(strcat('local MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(r)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

axes(ha(3))
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
title(strcat('local MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(s)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_MSST_win',num2str(winlen));
saveas(gcf, fname);

%% RS
data_rsh=data_reshape.';
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
spc_RS=[];
for i=1:bz
    spc_RS(i,:,:)  = RS(ret(i,:).',winlen);
end

if flag==1
    iter=bz-1;
else
    iter=bz;
end
ret=[];
for i=1:iter
    tmp=squeeze(spc_RS(i,:,:)).';
    ret=[ret;tmp(overlay/2+1:siglen-overlay/2,:)];
end
if flag==1
    tmp=squeeze(spc_RS(iter+1,:,:)).';
    tmp=tmp(end-left_len+1:end,:);
    ret=[ret;tmp(overlay/2+1:end,:)];
end

spc_RS=ret.';
spc_RS=abs(spc_RS);

tt=t;
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1))
clims = [max(max(20*log10((abs(spc_RS)))))-clims_lim,max(max(20*log10((abs(spc_RS)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('RS, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(t)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2))
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
title(strcat('RS, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(u)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_RS_win',num2str(winlen));
saveas(gcf, fname);

axes(ha(3))
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
title(strcat('RS, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(v)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_RS_win',num2str(winlen));
saveas(gcf, fname);

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
h=figure();
set(h,'position',[100 100 1200 400]);
ha=tight_subplot(1,3,[0.08 0.06],[.25 .08],[.06 .02]);
axes(ha(1));
ret=(10.^(ret/20)-1)/10;
clims = [max(max(20*log10((abs(ret)))))-clims_lim,max(max(20*log10((abs(ret)))))];
imagesc(t,yaxis,20*log10(fftshift(abs(ret.'),1)),clims)
set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / sec';'(w)'})
ylabel('Freq. / kHz')
title('TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',3);

axes(ha(2))
imagesc(t,yaxis,20*log10(fftshift(abs(ret.'),1)),clims)
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
title(strcat('local TFA-Net'))
xlabel({'Time / sec';'(x)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);

axes(ha(3))
imagesc(t,yaxis,20*log10(fftshift(abs(ret.'),1)),clims)
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','#D2691E','Linewidth',7);
set(gca,'ydir','normal')
set(gca,'xlim',[x1 x2],'ylim',[y1 y2]);
title(strcat('local TFA-Net'))
xlabel({'Time / sec';'(y)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_voice_TFA');
saveas(gcf, fname);














