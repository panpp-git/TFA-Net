clc
clear
close all
N=256;
nfft=256;
fsz=23;
winlen=32;
divide_times=1/2;
times_times=2;
x11=1.7; x22=2.5;
y11=-15;   y22=15;

x1=0.1;x2=1.29;
y1=25.5;y2=62.7;

ylim_low=-20;
%% SIMU1
fs=128;
ts=1/fs;
t = 0 : ts : 3-ts;

c1 = 2 * pi * 10;            % initial frequency of the chirp excitation
c2 = 2 * pi * 2/2;           % set the speed of frequency change be 1 Hz/second
c3 = 2 * pi * 1/10;
c4 = 2 * pi * -2/2;

d1 = -2 * pi * 10;            % initial frequency of the chirp excitation
d2 = 2 * pi * 2/2;           % set the speed of frequency change be 1 Hz/second
d3 = 2 * pi * 1/10;
d4 = 2 * pi * 2/2;

e1 = 2 * pi * 0;            % initial frequency of the chirp excitation
e2 = 2 * pi * 2;           % set the speed of frequency change be 1 Hz/second
e3 = 2 * pi * (-0.07);
e4 = 2 * pi * (-2.5);

Sig1 = exp(1i*(c1 * t + c4 * t.^4 /4));   % get the A(t)
Sig2 = exp(1i*(d1 * t + d4 * t.^4 /4));   % get the A(t)
Sig3 = 5*exp(1i*(e1 * t + e4 *sin(2.5*pi*t) ));   % get the A(t)
Sig=Sig1+Sig2+Sig3;
data_reshape=Sig.';
data_reshape=awgn(data_reshape,10);
ydelta=fs/nfft;
yaxis=(0:ydelta:fs-ydelta)-fs/2;
if1=(c1+c4 * t.^3)/2/pi;
if2=(d1+ d4 * t.^3)/2/pi;
if3=(e1+ e4 *cos(2.5*pi*t)*2.5*pi)/2/pi ;
figure
plot(t,if1)
hold on
plot(t,if2)
hold on
plot(t,if3)
ylim([ylim_low max(yaxis)])
xlabel({'Time / sec';'(a)'})
ylabel('Doppler / Hz')
title('Ground Truth')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_Ground_Truth');
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_Ground_Truth','.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% STFT
data_reshape1=[zeros(winlen/2,1);data_reshape;zeros(winlen/2-1,1)];
spc_STFT=abs(stft(data_reshape1,'Window',hamming(winlen).','OverlapLength',winlen-1,'FFTLength',nfft));
data_reshape2=[zeros(winlen*divide_times/2,1);data_reshape;zeros(winlen*divide_times/2-1,1)];
spc_STFT2=abs(stft(data_reshape2,'Window',hamming(winlen*divide_times).','OverlapLength',winlen*divide_times-1,'FFTLength',nfft));
data_reshape3=[zeros(winlen*times_times/2,1);data_reshape;zeros(winlen*times_times/2-1,1)];
spc_STFT3=abs(stft(data_reshape3,'Window',hamming(winlen*times_times).','OverlapLength',winlen*times_times-1,'FFTLength',nfft));
tt=t;
figure;
imagesc(tt,yaxis,((abs(spc_STFT2))));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
title(strcat('STFT, winlen=',num2str(winlen*divide_times)))
xlabel({'Time / sec';'(c)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(abs(spc_STFT2)))
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_STFT_win',num2str(winlen*divide_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_STFT_win',num2str(winlen*divide_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

figure;
imagesc(tt,yaxis,((abs(spc_STFT3))));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
title(strcat('STFT, winlen=',num2str(winlen*times_times)))
xlabel({'Time / sec';'(d)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(abs(spc_STFT3)))
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_STFT_win',num2str(winlen*times_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_STFT_win',num2str(winlen*times_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

figure;
imagesc(tt,yaxis,((abs(spc_STFT))));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
title(strcat('STFT, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(e)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(abs(spc_STFT)))
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_STFT_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_STFT_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);
%% SST
spc_SST  = SST2(data_reshape,winlen);
spc_SST2  = SST2(data_reshape,winlen*divide_times);
spc_SST3  = SST2(data_reshape,winlen*times_times);
spc_SST=abs(spc_SST);

tt=t;
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SST2),1)));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
xlabel({'Time / sec';'(f)'})
ylabel('Freq. / Hz')
title(strcat('SST, winlen=',num2str(winlen*divide_times)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_SST2),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SST_win',num2str(winlen*divide_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SST_win',num2str(winlen*divide_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SST3),1)));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
xlabel({'Time / sec';'(g)'})
ylabel('Freq. / Hz')
title(strcat('SST, winlen=',num2str(winlen*times_times)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_SST3),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SST_win',num2str(winlen*times_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SST_win',num2str(winlen*times_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
xlabel({'Time / sec';'(h)'})
ylabel('Freq. / Hz')
title(strcat('SST, winlen=',num2str(winlen)))
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_SST),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SST_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SST_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% SET
[spc_SET,tfr]  = SET_Y2(data_reshape,winlen);
[spc_SET2,tfr]  = SET_Y2(data_reshape,winlen*divide_times);
[spc_SET3,tfr]  = SET_Y2(data_reshape,winlen*times_times);
spc_SET=abs(spc_SET);

tt=t;

figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SET2),1)));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
title(strcat('SET, winlen=',num2str(winlen*divide_times)))
xlabel({'Time / sec';'(i)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_SET2),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SET_win',num2str(winlen*divide_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SET_win',num2str(winlen*divide_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SET3),1)));
ylim([ylim_low max(yaxis)])
title(strcat('SET, winlen=',num2str(winlen*times_times)))
set(gca,'ydir','normal')
xlabel({'Time / sec';'(j)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_SET3),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SET_win',num2str(winlen*times_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SET_win',num2str(winlen*times_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

figure;
imagesc(tt,yaxis,(fftshift(abs(spc_SET),1)));
ylim([ylim_low max(yaxis)])
title(strcat('SET, winlen=',num2str(winlen)))
set(gca,'ydir','normal')
xlabel({'Time / sec';'(k)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_SET),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SET_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_SET_win',num2str(winlen),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

%% MSST
[spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
[spc_MSST2,tfr,omega2] = MSST_Y_new2(data_reshape,winlen*divide_times,3);
[spc_MSST3,tfr,omega2] = MSST_Y_new2(data_reshape,winlen*times_times,3);
spc_MSST=spc_MSST1;

tt=t;
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_MSST2),1)));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
title(strcat('MSST, winlen=',num2str(winlen*divide_times)))
xlabel({'Time / sec';'(l)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_MSST2),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_MSST_win',num2str(winlen*divide_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_MSST_win',num2str(winlen*divide_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);
figure;
imagesc(tt,yaxis,(fftshift(abs(spc_MSST3),1)));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
title(strcat('MSST, winlen=',num2str(winlen*times_times)))
xlabel({'Time / sec';'(m)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_MSST3),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_MSST_win',num2str(winlen*times_times));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_MSST_win',num2str(winlen*times_times),'.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

figure;
imagesc(tt,yaxis,(fftshift(abs(spc_MSST),1)));
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
title(strcat('MSST, winlen=',num2str(winlen)))
xlabel({'Time / sec';'(n)'})
ylabel('Freq. / Hz')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,(fftshift(abs(spc_MSST),1)));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_MSST_win',num2str(winlen));
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_MSST_win',num2str(winlen),'.pdf');
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
imagesc(t,yaxis,fftshift(abs(ret.'),1))
ylim([ylim_low max(yaxis)])
set(gca,'ydir','normal')
xlabel({'Time / sec';'(b)'})
ylabel('Freq. / Hz')
title('TFA-Net')
set(gca,'FontSize',fsz); 
set(get(gca,'XLabel'),'FontSize',fsz);
set(get(gca,'YLabel'),'FontSize',fsz);
rectangle('Position',[x11 y11 x22-x11 y22-y11],'EdgeColor','red','Linewidth',3);
rectangle('Position',[x1 y1 x2-x1 y2-y1],'EdgeColor','red','Linewidth',3);
line([x11,x2],[y22,y1],'color','r','Linewidth',2,'LineStyle','-.')
line([x22,x2],[y22,y2],'color','r','Linewidth',2,'LineStyle','-.')
axes('Position',[0.2,0.62,0.285,0.25]);
imagesc(tt,yaxis,fftshift(abs(ret.'),1));
set(gca,'ydir','normal')
set(gca,'xlim',[x11 x22],'ylim',[y11 y22]);
fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_TFA-Net');
saveas(gcf, fname);
% fname=strcat('F:\pycharm_proj\cResTF\TFA_Net\figures_TFA_submit_20220120\figs\','exp1_simu2_TFA-Net','.pdf');
% export_fig(gcf , '-eps' , '-r300' , '-painters' , fname);

