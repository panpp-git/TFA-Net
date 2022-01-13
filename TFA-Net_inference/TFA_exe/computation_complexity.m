clc
clear
close all
N=256;
nfft=256;
fsz=16;
winlen=64;
divide_times=1/2;
times_times=2;
x11=0.3; x22=2.7;
y11=-1;   y22=20;

x1=0.45+5.555;x2=4.4+5.555;
y1=-48.5;y2=-4;
%% SIMU1
fs=128;
FS=fs;
ts=1/fs;
t = 0:ts:2-ts;
Sig1 = exp(1i*2*pi*(8* t + 6 *sin(t)));   % get the A(t)
Sig2 = exp(1i*2*pi*(10 * t + 6 *sin(1.5*t) ));   % get the A(t)
Sig=Sig1+Sig2;
data_reshape=Sig.';

ITER=1000;
time_STFT=zeros(1,ITER);
time_SST=zeros(1,ITER);
time_SET=zeros(1,ITER);
time_MSST=zeros(1,ITER);
time_TFA=zeros(1,ITER);
time_RS=zeros(1,ITER);
time_WVD=zeros(1,ITER);
time_G=zeros(1,ITER);
for i=1:ITER
    i
    %% STFT
    data_reshape1=[zeros(winlen/2,1);data_reshape;zeros(winlen/2-1,1)];
    tic
    spc_STFT=abs(stft(data_reshape1,'Window',hamming(winlen).','OverlapLength',winlen-1,'FFTLength',nfft));
    aa=toc;
    time_STFT(i)=aa;
    %% SST
    tic
    spc_SST  = SST2(data_reshape,winlen);
    aa=toc;
    time_SST(i)=aa;
    %% SET
    tic
    [spc_SET,~]  = SET_Y2(data_reshape,winlen);
    aa=toc;
    time_SET(i)=aa;
    %% MSST
    tic
    [spc_MSST1,tfr,omega2] = MSST_Y_new2(data_reshape,winlen,3);
    aa=toc;
    time_MSST(i)=aa;
    %% ResFreq
    data_rsh=data_reshape.';
    mv=max(abs(data_rsh));
    ret=data_rsh/mv;
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
    tic
    net_flag=system('curl -s 127.0.0.1:5012/');
    aa=toc;
    time_TFA(i)=aa;
    %% GWarblet
    tic
    [Spec,f] = GWarblet(data_reshape,FS,0,1,nfft,winlen);
    [v l] = max(Spec,[],1);
    [IF, a_n,b_n,fm] = get_fscoeff(f(l),length(t),t,FS);
    WinLen =winlen*4;
    [Spec,f] = GWarblet(data_reshape,FS,[-a_n;b_n],fm(2:end),nfft,WinLen);
    aa=toc;
    time_G(i)=aa;
    %% WVD
    tic
    Spec=wvd(data_reshape,'smoothedPseudo','NumFrequencyPoints',nfft);
    aa=toc;
    time_WVD(i)=aa;
    %% RS
    tic
    spc_RS  = RS(data_reshape,winlen);
    aa=toc;
    time_RS(i)=aa;
end

time=[mean(time_STFT),mean(time_WVD),mean(time_G),mean(time_RS),mean(time_SST),mean(time_SET),mean(time_MSST),mean(time_TFA)];
save time_cost.mat time
