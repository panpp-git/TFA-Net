
clc
clear
close all
nfft=256;
fsz=16;
clims_lim=30;


load('batdata2.mat');
SampFreq = 1000000/7;
FS=SampFreq;
n=length(data);
data_reshape=data;
time=(1:n)/SampFreq;
t=time*1000;
ydelta=FS/nfft;
yaxis=((0:ydelta:FS-ydelta)-FS/2)/1000;
winlen=48;
ylim_low=0;

x11=14.7; x22=16.9;
y11=0;   y22=20;

x1=19.1;x2=23.4;
y1=23.6;y2=43;

%% RS
spc_RS  = RS(data_reshape,winlen);
spc_RS=abs(spc_RS);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_RS)))))-clims_lim,max(max(20*log10((abs(spc_RS)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_RS),1)),clims);
% set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('RS, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(d)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_RS_win',num2str(winlen));
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
% set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('GWarblet'))
xlabel({'Time / sec';'(c)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_GWarblet');
saveas(gcf, fname);

%% WVD
yaxis1=(0:ydelta:FS/2-ydelta)/1000;
Spec=wvd(data_reshape,'smoothedPseudo','NumFrequencyPoints',nfft);
figure;
clims = [max(max(20*log10((abs(Spec)))))-40,max(max(20*log10((abs(Spec)))))];
imagesc(t,yaxis1,20*log10((abs(Spec))),clims);
% set(gca,'ydir','normal')
% ylim([ylim_low max(yaxis)])
title(strcat('SPWVD'))
xlabel({'Time / sec';'(b)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_SPWVD');
saveas(gcf, fname);

%% cwt
% Spec=cwt((data_reshape));
% Spec=flipud((Spec));
% figure;
% clims = [max(max(20*log10((abs(Spec)))))-clims_lim,max(max(20*log10((abs(Spec)))))];
% imagesc(t,yaxis,20*log10((abs(Spec))),clims);
% % set(gca,'ydir','normal')
% ylim([ylim_low max(yaxis)])
% title(strcat('CWT'))
% xlabel({'Time / sec';'(c)'})
% ylabel('Freq. / kHz')
% set(gca,'FontSize',fsz); 
% set(get(gca,'XLabel'),'FontSize',fsz);
% set(get(gca,'YLabel'),'FontSize',fsz);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_CWT');
% saveas(gcf, fname);
%% STFT
data_reshape1=[zeros(winlen/2,1);data_reshape;zeros(winlen/2-1,1)];
spc_STFT=abs(stft(data_reshape1,'Window',hamming(winlen).','OverlapLength',winlen-1,'FFTLength',nfft));
tt=t;

figure;
clims = [max(max(20*log10((abs(spc_STFT)))))-clims_lim,max(max(20*log10((abs(spc_STFT)))))];
imagesc(tt,yaxis,20*log10((abs(spc_STFT))),clims);
% set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(a)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_STFT_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_STFT_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);
%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST=abs(spc_SST);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_SST)))))-clims_lim,max(max(20*log10((abs(spc_SST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SST),1)),clims);
% set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / sec';'(e)'})
ylabel('Freq. / kHz')
title(strcat('SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_SST_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_SST_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% SET
[spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
spc_SET=abs(spc_SET);
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_SET)))))-clims_lim,max(max(20*log10((abs(spc_SET)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_SET),1)),clims);
% set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('SET, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(f)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_SET_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_SET_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% MSST
[spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
spc_MSST=spc_MSST1;
tt=t;
figure;
clims = [max(max(20*log10((abs(spc_MSST)))))-clims_lim,max(max(20*log10((abs(spc_MSST)))))];
imagesc(tt,yaxis,20*log10(fftshift(abs(spc_MSST),1)),clims);
% set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
title(strcat('MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(g)'})
ylabel('Freq. / kHz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_MSST_win',num2str(winlen));
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
imagesc(t,yaxis,(20*log10(fftshift(abs(ret.'),1))),clims)
% % set(gca,'ydir','normal')
ylim([ylim_low max(yaxis)])
xlabel({'Time / sec';'(i)'})
ylabel('Freq. / kHz')
title('TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_bat_TFA-Net');
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp2_heart_TFA-Net','.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);