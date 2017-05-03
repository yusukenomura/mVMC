/*
mVMC - A numerical solver package for a wide range of quantum lattice models based on many-variable Variational Monte Carlo method
Copyright (C) 2016 Takahiro Misawa, Satoshi Morita, Takahiro Ohgoe, Kota Ido, Mitsuaki Kawamura, Takeo Kato, Masatoshi Imada.

This program is developed based on the mVMC-mini program
(https://github.com/fiber-miniapp/mVMC-mini)
which follows "The BSD 3-Clause License".

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details. 

You should have received a copy of the GNU General Public License 
along with this program. If not, see http://www.gnu.org/licenses/. 
*/
/*-------------------------------------------------------------
 * Variational Monte Carlo
 * calculate physical quantities
 *-------------------------------------------------------------
 * by Satoshi Morita 
 *-------------------------------------------------------------*/

void VMCMainCal(MPI_Comm comm);
void clearPhysQuantity();
void calculateOptTransDiff(double complex *srOptO, const double complex ipAll);
void calculateOO_matvec(double complex *srOptOO, double complex *srOptHO, const double complex *srOptO,
                 const double complex w, const double complex e, const int srOptSize);
void calculateOO(double complex *srOptOO, double complex *srOptHO, const double complex *srOptO,
                 const double  w, const double complex e, const int srOptSize);
void calculateOO_real(double *srOptOO, double *srOptHO, const double *srOptO,
                 const double w, const double e, const int srOptSize);
void calculateOO_Store_real(double *srOptOO_real, double *srOptHO_real,  double *srOptO_real,
                 const double w, const double e,  int srOptSize, int sampleSize);
void calculateOO_Store(double complex *srOptOO, double complex *srOptHO,  double complex *srOptO,
                 const double w, const double complex e,  int srOptSize, int sampleSize);
void calculateQQQQ(double *qqqq, const double *lslq, const double w, const int nLSHam);
void calculateQCAQ(double *qcaq, const double *lslca, const double *lslq,
                   const double w, const int nLSHam, const int nCA);
void calculateQCACAQ(double *qcacaq, const double *lslca, const double w,
                     const int nLSHam, const int nCA, const int nCACA,
                     int **cacaIdx);

void VMCMainCal(MPI_Comm comm) {
  int *eleIdx,*eleCfg,*eleNum,*eleProjCnt;
/* added by YN */ 
  int *hiddenCfg1,*tmpHiddenCfg1;
  int *hiddenCfg2,*tmpHiddenCfg2;
  int samplehidden,f,j,offset,idx,rsi;
  double complex *thetaHidden1,*tmpTheta1; /* modified by KI */
  double complex *thetaHidden2,*tmpTheta2; /* modified by KI */
  double x,y1,y2;  
  int nFail = 0; 
/* added by YN */ 
  double complex we,ip; /* modified by YN */
  double complex e[2*NVMCSampleHidden]; 
  double w;
  double sqrtw; /* modified by YN */

  const int qpStart=0;
  const int qpEnd=NQPFull;
  int sample,sampleStart,sampleEnd,sampleSize;
  int i,info,tmp_i;

  /* optimazation for Kei */
  const int nProj=NProj;
  /* added by YN */
  const int offset1=2*NProj+2*NHiddenVariable; 
  const int offset2=2*NProj+2*NHiddenVariable+2*NSlater; 
  const int nSizeTheta=NSizeTheta;
  const int nSetHidden=NSetHidden;
  const int nNeuronSample=NNeuronSample;
  const int nVMCSampleHidden=NVMCSampleHidden;
  const int nVMCSampleHidden2=2*NVMCSampleHidden;
  const int nIntPerNeuron=NIntPerNeuron;
  const int nNeuronPerSet=NNeuronPerSet;
  /* added by YN */
  double complex *srOptO = SROptO;
  double         *srOptO_real = SROptO_real;

  int rank,size,int_i,int_j; /* modified by YN */
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
#ifdef _DEBUG
  printf("  Debug: SplitLoop\n");
#endif
  SplitLoop(&sampleStart,&sampleEnd,NVMCSample,rank,size);

  /* initialization */
  StartTimer(24);
  clearPhysQuantity();
  StopTimer(24);
  for(sample=sampleStart;sample<sampleEnd;sample++) {
    eleIdx = EleIdx + sample*Nsize;
    eleCfg = EleCfg + sample*Nsite2;
    eleNum = EleNum + sample*Nsite2;
    eleProjCnt = EleProjCnt + sample*NProj;
    hiddenCfg1 = HiddenCfg1 + sample*NSizeHiddenCfgSave; /* added by YN */
    hiddenCfg2 = HiddenCfg2 + sample*NSizeHiddenCfgSave; /* added by YN */
    thetaHidden1 = ThetaHidden1 + sample*NSizeThetaSave; /* added by YN */
    thetaHidden2 = ThetaHidden2 + sample*NSizeThetaSave; /* added by YN */
//DEBUG
    /* for(i=0;i<Nsite;i++) {
      printf("Debug: sample=%d: i=%d  up=%d down =%d \n",sample,i,eleCfg[i+0*Nsite],eleCfg[i+1*Nsite]);
      }*/
//DEBUG

    StartTimer(40);
#ifdef _DEBUG
    printf("  Debug: sample=%d: CalculateMAll \n",sample);
#endif
    if(iComplexFlgOrbital==0){ /* modified by YN */ /* Warning !! Temporal Treatment */
       info = CalculateMAll_real(eleIdx,qpStart,qpEnd); // InvM_real,PfM_real will change
       #pragma omp parallel for default(shared) private(tmp_i)
       for(tmp_i=0;tmp_i<NQPFull*(Nsize*Nsize+1);tmp_i++)  InvM[tmp_i]= InvM_real[tmp_i]; // InvM will be used in  SlaterElmDiff_fcmp
       /* added by YN */ /* Warning!! Temporal Treatment */
       #pragma omp parallel for default(shared) private(tmp_i)
       for(tmp_i=0;tmp_i<qpEnd-qpStart;tmp_i++)  PfM[tmp_i]= PfM_real[tmp_i]; 
       /* added by YN */ /* Warning!! Temporal Treatment */
    }else{
      info = CalculateMAll_fcmp(eleIdx,qpStart,qpEnd); // InvM,PfM will change
    }
    StopTimer(40);

    if(info!=0) {
      fprintf(stderr,"warning: VMCMainCal rank:%d sample:%d info:%d (CalculateMAll)\n",rank,sample,info);
      continue;
    }
#ifdef _DEBUG
    printf("  Debug: sample=%d: CalculateIP \n",sample);
#endif
    if(AllComplexFlag==0){
      ip = CalculateIP_real(PfM_real,qpStart,qpEnd,MPI_COMM_SELF);
    }else{
      ip = CalculateIP_fcmp(PfM,qpStart,qpEnd,MPI_COMM_SELF);
    } 

    //x = LogProjVal(eleProjCnt);
#ifdef _DEBUG
    printf("  Debug: sample=%d: LogProjVal \n",sample);
#endif
    /* modified by YN */
    x = LogProjVal(eleProjCnt);
    y1 = LogHiddenWeightVal(thetaHidden1);
    y2 = LogHiddenWeightVal(thetaHidden2);
    /* calculate reweight */
    w = 2.0*(log(fabs(ip))+x) + y1 + y2 - logSqPfFullSlater[sample];
    if( fabs(w) > 0.0001 ){
      if( fabs(w) > 1.0 && sample == sampleStart) printf("warning: VMCMainCal rank:%d sample:%d difference=%e\n",rank,sample,w);
      nFail++; 
    } 
    /* modified by YN */
    w =1.0;
#ifdef _DEBUG
    printf("  Debug: sample=%d: isfinite \n",sample);
#endif
    if( !isfinite(w) ) {
      fprintf(stderr,"warning: VMCMainCal rank:%d sample:%d w=%e\n",rank,sample,w);
      continue;
    }

    StartTimer(41);
    /* calculate energy */
#ifdef _DEBUG
    printf("  Debug: sample=%d: calculateHam \n",sample);
#endif
    if(AllComplexFlag==0){
#ifdef _DEBUG
      printf("  Debug: sample=%d: calculateHam_real \n",sample);
#endif
      CalculateHamiltonian_real(e,creal(ip),eleIdx,eleCfg,eleNum,eleProjCnt, /* modified by YN */
                                hiddenCfg1,hiddenCfg2,thetaHidden1,thetaHidden2); /* added by YN */
    }else{
#ifdef _DEBUG
      printf("  Debug: sample=%d: calculateHam_cmp \n",sample);
#endif
      CalculateHamiltonian(e,ip,eleIdx,eleCfg,eleNum,eleProjCnt, /* modified by YN */
                           hiddenCfg1,hiddenCfg2,thetaHidden1,thetaHidden2); /* added by YN */
    }
    //printf("DEBUG: rank=%d: sample=%d ip= %lf %lf\n",rank,sample,creal(ip),cimag(ip));
    StopTimer(41);
    /* modified by YN */
    for(i=0;i<nVMCSampleHidden2;i++){
      if( !isfinite(creal(e[i]) + cimag(e[i])) ) {
        fprintf(stderr,"warning: VMCMainCal rank:%d sample:%d e=%e\n",rank,sample,creal(e[i])); //TBC
        continue;
      }
    }

    Wc += w;
    for(i=0;i<nVMCSampleHidden2;i++){
      Etot  += w * e[i];
      Etot2 += w * conj(e[i]) * e[i];
    }
    /* modified by YN */
#ifdef _DEBUG
    printf("  Debug: sample=%d: calculateOpt \n",sample);
#endif
    if(NVMCCalMode==0) {
      /* Calculate O for correlation fauctors */
      srOptO[0] = 1.0+0.0*I;//   real 
      srOptO[1] = 0.0+0.0*I;//   real 
      #pragma loop noalias
      for(i=0;i<nProj;i++){ 
        srOptO[(i+1)*2]     = (double)(eleProjCnt[i]); // even real
        srOptO[(i+1)*2+1]   = 0.0+0.0*I;               // odd  comp
      }

      /* added by YN */
      StartTimer(74);
      /* Hidden-layer magnetic field 
         This part assumes that the magnetic field is uniform for each set of Hidden variables. 
         In this case, NHiddenMagField = NSetHidden */
      tmp_i = nProj; 
      // #pragma loop noalias  /* comment by YN: is this line needed? */
      for(f=0;f<nSetHidden;f++){ 
        x = 0.0;
        for(samplehidden=0;samplehidden<nVMCSampleHidden;samplehidden++){
          tmpTheta1 = thetaHidden1 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
          tmpTheta2 = thetaHidden2 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
          /* change */
          tmpHiddenCfg1 = hiddenCfg1 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
          tmpHiddenCfg2 = hiddenCfg2 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
          for(i=0;i<nNeuronPerSet;i++) x += (double)(tmpHiddenCfg1[i]);//cTanh(tmpTheta1[i]);  /* modified by KI */
          for(i=0;i<nNeuronPerSet;i++) x += (double)(tmpHiddenCfg2[i]);//cTanh(tmpTheta2[i]);  /* modified by KI */
          /* change */
        }
        x /= 2.0*(double)(nVMCSampleHidden);
        srOptO[(tmp_i+1)*2]   = x;         // even real
        srOptO[(tmp_i+1)*2+1] = x*I;       // odd  comp   /* modified by KI */
        tmp_i++;
      }

      /* Interaction between hidden and phyisical variables 
         j-th type of interaction in f-th set connects i-th neuron with 
         HiddenPhysIntIdx2[f*NIntPerNeuron+j][i]-th physical variable. */
      // #pragma loop noalias  /* comment by YN: is this line needed? */
      for(f=0;f<nSetHidden;f++){ 
        offset = f*NIntPerNeuron;
        for(j=0;j<nIntPerNeuron;j++) {
          idx = offset + j; 
          x = 0.0;
          for(samplehidden=0;samplehidden<nVMCSampleHidden;samplehidden++){
            tmpTheta1 = thetaHidden1 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
            tmpTheta2 = thetaHidden2 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
            /* change */
            tmpHiddenCfg1 = hiddenCfg1 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
            tmpHiddenCfg2 = hiddenCfg2 + f*nNeuronPerSet + samplehidden*nSizeTheta; 
            for(i=0;i<nNeuronPerSet;i++) {
             rsi = HiddenPhysIntIdx2[idx][i]; 
             //x += cTanh(tmpTheta1[i])*(double complex)(2*eleNum[rsi]-1);  /* modified by KI */
             //x += cTanh(tmpTheta2[i])*(double complex)(2*eleNum[rsi]-1);  /* modified by KI */
             x += (double)(tmpHiddenCfg1[i])*(double)(2*eleNum[rsi]-1);  /* modified by KI */
             x += (double)(tmpHiddenCfg2[i])*(double)(2*eleNum[rsi]-1);  /* modified by KI */
            /* change */
            }
          }
          x /= 2.0*(double)(nVMCSampleHidden);
          srOptO[(tmp_i+1)*2]   = x;               // even real
          srOptO[(tmp_i+1)*2+1] = x*I;       // odd  comp  /* modified by KI */
          tmp_i++;
        }
      }
      if( 2*tmp_i != offset1 ) {
        fprintf(stderr, " 2*tmp_i != offset1 \n");
        MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
      }
      StopTimer(74);
      /* added by YN */

      StartTimer(42);
      /* SlaterElmDiff */
      SlaterElmDiff_fcmp(SROptO+offset1+2,ip,eleIdx); //TBC: using InvM not InvM_real /* modified by YN */
      StopTimer(42);
      
      if(FlagOptTrans>0) { // this part will be not used
        calculateOptTransDiff(SROptO+offset2+2, ip); //TBC /* modified by YN */
      }
//[s] this part will be used for real varaibles
      if(AllComplexFlag==0){
        #pragma loop noalias
        for(i=0;i<SROptSize;i++){ 
          srOptO_real[i] = creal(srOptO[2*i]);       
        }
      }
//[e]

      StartTimer(43);
      /* Calculate OO and HO */
      if(NStoreO==0){
        //calculateOO_matvec(SROptOO,SROptHO,SROptO,w,e,SROptSize);
        if(AllComplexFlag==0){
          calculateOO_real(SROptOO_real,SROptHO_real,SROptO_real,w,creal(e),SROptSize);
        }else{
          calculateOO(SROptOO,SROptHO,SROptO,w,e,SROptSize);
        } 
      }else{
        we    = w*e;
        sqrtw = sqrt(w); 
        if(AllComplexFlag==0){
          #pragma omp parallel for default(shared) private(int_i)
          for(int_i=0;int_i<SROptSize;int_i++){
            // SROptO_Store for fortran
            SROptO_Store_real[int_i+sample*SROptSize]  = sqrtw*SROptO_real[int_i];
            SROptHO_real[int_i]                       += creal(we)*SROptO_real[int_i]; 
          }
        }else{
          /* modified by YN */ /* Warning!! Temporal Treatment */
          // SROptO_Store for fortran
          SROptO_Store[  sample*(SROptSmatDim+2)]  = sqrtw*SROptO[0]; 
          SROptO_Store[1+sample*(SROptSmatDim+2)]  = sqrtw*SROptO[1]; 
          SROptHO[0] += we*SROptO[0]; 
          SROptHO[1] += we*SROptO[1]; 
          #pragma omp parallel for default(shared) private(int_i,int_j) 
          for(int_i=0;int_i<SROptSmatDim;int_i++){
            int_j = SmatIdxtoParaIdx[int_i]; 
            if( int_j >= 2*NPara ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE); 
            SROptO_Store[(int_i+2)+sample*(SROptSmatDim+2)]  = sqrtw*SROptO[(int_j+2)]; 
            SROptHO[int_i+2]                                += we*SROptO[(int_j+2)]; 
          }
          /* modified by YN */ /* Warning!! Temporal Treatment */
        }
      } 
      StopTimer(43);

    } else if(NVMCCalMode==1) {
      StartTimer(42);
      /* Calculate Green Function */
      CalculateGreenFunc(w,ip,eleIdx,eleCfg,eleNum,eleProjCnt, /* modified by YN */
                         hiddenCfg1,hiddenCfg2,thetaHidden1,thetaHidden2); /* modified by YN */
      StopTimer(42);

      if(NLanczosMode>0){
        // ignoring Lanczos: to be added
        /* Calculate local QQQQ */
        //StartTimer(43);
        //LSLocalQ(e,ip,eleIdx,eleCfg,eleNum,eleProjCnt);
        //calculateQQQQ(QQQQ,LSLQ,w,NLSHam);
        //StopTimer(43);
        //if(NLanczosMode>1){
          /* Calculate local QcisAjsQ */
          //StartTimer(44);
          //LSLocalCisAjs(e,ip,eleIdx,eleCfg,eleNum,eleProjCnt);
          //calculateQCAQ(QCisAjsQ,LSLCisAjs,LSLQ,w,NLSHam,NCisAjs);
          //calculateQCACAQ(QCisAjsCktAltQ,LSLCisAjs,w,NLSHam,NCisAjs,
          //                NCisAjsCktAlt,CisAjsCktAltIdx);
          //StopTimer(44);
        //}
      }
    }
  } /* end of for(sample) */
  /* added by YN */
  if( nFail > (sampleEnd-sampleStart)/10 ){
    NPfUpdate /= 2;  
    if( NPfUpdate < 10 ) NPfUpdate = 10; 
    if( nFail > (sampleEnd-sampleStart)/4 ) printf("warning: VMCMainCal rank: %d nFail= %d NPfUpdate= %d\n",rank,nFail,NPfUpdate); 
  } else if( nFail < (sampleEnd-sampleStart)/40 ) {
    NPfUpdate *= 2;  
    if( NPfUpdate > NPfUpdate0 ) NPfUpdate = NPfUpdate0; 
  }
  /* added by YN */
// calculate OO and HO at NVMCCalMode==0
  if(NStoreO!=0 && NVMCCalMode==0){
    sampleSize=sampleEnd-sampleStart;
    if(AllComplexFlag==0){
      StartTimer(45);
      calculateOO_Store_real(SROptOO_real,SROptHO_real,SROptO_Store_real,creal(w),creal(e),SROptSize,sampleSize);
      StopTimer(45);
    }else{
      StartTimer(45);
      calculateOO_Store(SROptOO,SROptHO,SROptO_Store,w,e,SROptSmatDim+2,sampleSize); /* modified by YN */ /* Warning!! Temporal Treatment */
      StopTimer(45);
    }
  }
  return;
}

void clearPhysQuantity(){
  int i,n;
  double complex *vec;
  double  *vec_real;
  Wc = Etot = Etot2 = 0.0;
  if(NVMCCalMode==0) {
    /* SROptOO, SROptHO, SROptO */
    n = (SROptSmatDim*AllComplexFlag+2)*(SROptSmatDim*AllComplexFlag+2) + 4*SROptSize; // TBC /* modified by YN */ /* Warning!! Temporal Treatment */  
    vec = SROptOO;
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<n;i++) vec[i] = 0.0+0.0*I;
// only for real variables
    if(AllComplexFlag==0){ /* added by YN */
    n = (SROptSize)*(SROptSize+2); // TBC
    vec_real = SROptOO_real;
    #pragma omp parallel for default(shared) private(i)
    for(i=0;i<n;i++) vec_real[i] = 0.0;
    } /* added by YN */
  } else if(NVMCCalMode==1) {
    /* CisAjs, CisAjsCktAlt, CisAjsCktAltDC */
    n = 2*NCisAjs+NCisAjsCktAlt+NCisAjsCktAltDC;
    vec = PhysCisAjs;
    for(i=0;i<n;i++) vec[i] = 0.0;
    if(NLanczosMode>0) {
      /* QQQQ, LSLQ */
      n = NLSHam*NLSHam*NLSHam*NLSHam + NLSHam*NLSHam;
      vec = QQQQ;
      for(i=0;i<n;i++) vec[i] = 0.0;
      if(NLanczosMode>1) {
        /* QCisAjsQ, QCisAjsCktAltQ, LSLCisAjs */
        n = NLSHam*NLSHam*NCisAjs + NLSHam*NLSHam*NCisAjsCktAlt
          + NLSHam*NCisAjs;
        vec = QCisAjsQ;
        for(i=0;i<n;i++) vec[i] = 0.0;
      }
    }
  }
  return;
}

void calculateOptTransDiff(double complex *srOptO, const double complex ipAll) {
  int i,j;
  double complex ip;
  double complex *pfM;

  for(i=0;i<NQPOptTrans;++i) {
    ip = 0.0;
    pfM = PfM + i*NQPFix;
    for(j=0;j<NQPFix;++j) {
      ip += QPFixWeight[j] * pfM[j];
    }
    srOptO[i] = ip/ipAll;
  }

  return;
}

void calculateOO_Store_real(double *srOptOO_real, double *srOptHO_real, double *srOptO_Store_real,
                 const double w, const double e, int srOptSize, int sampleSize) {

//#define M_DGEM dgemm_

extern int
dgemm_(char *jobz, char *uplo, int *m, int *n, int *k, double *alpha, double *a, int *lda, double *b, int *ldb,
       double *beta, double *c, int *ldc);

  char jobz, uplo;
  double alpha,beta;
  
  alpha = 1.0;
  beta  = 0.0;
  
  jobz = 'N';
  uplo = 'T';
  dgemm_(&jobz,&uplo,&srOptSize,&srOptSize,&sampleSize,&alpha,srOptO_Store_real,&srOptSize,srOptO_Store_real,&srOptSize,&beta,srOptOO_real,&srOptSize);

  return;
}



void calculateOO_Store(double complex *srOptOO, double complex *srOptHO, double complex *srOptO_Store,
                 const double w, const double complex e, int srOptSize, int sampleSize) {

  //#define M_DGEM dgemm_

  extern int zgemm_(char *jobz, char *uplo, int *m,int *n,int *k,double complex *alpha,  double complex *a, int *lda, double complex *b, int *ldb,
                    double complex *beta,double complex *c,int *ldc);

  char jobz, uplo;
  double complex alpha,beta;
  
  alpha = 1.0;
  beta  = 0.0;
  
  jobz = 'N';
  uplo = 'C';
  zgemm_(&jobz,&uplo,&srOptSize,&srOptSize,&sampleSize,&alpha,srOptO_Store,&srOptSize,srOptO_Store,&srOptSize,&beta,srOptOO,&srOptSize);

  return;
}




//void calculateOO(double complex *srOptOO, double complex *srOptHO, const double complex *srOptO,
//                 const double w, const double complex e, const int srOptSize) {
//  double we=w*e;
//
//  #define M_DAXPY daxpy_
//  #define M_DGER dger_
//
//  extern int M_DAXPY(const int *n, const double *alpha, const double *x, const int *incx,
//                     double *y, const int *incy);
//  extern int M_DGER(const int *m, const int *n, const double *alpha,
//                    const double *x, const int *incx, const double *y, const int *incy, 
//                    double *a, const int *lda);
//  int m,n,incx,incy,lda;
//  m=n=lda=srOptSize;
//  incx=incy=1;
//
//  /* OO[i][j] += w*O[i]*O[j] */
//  M_DGER(&m, &n, &w, srOptO, &incx, srOptO, &incy, srOptOO, &lda);
//
//  /* HO[i] += w*e*O[i] */
//  M_DAXPY(&n, &we, srOptO, &incx, srOptHO, &incy);
//
//  return;
//}

void calculateOO_matvec(double complex *srOptOO, double complex *srOptHO, const double complex *srOptO,
                 const double complex w, const double complex e, const int srOptSize) {
  double complex we=w*e;

  #define M_ZAXPY zaxpy_
  #define M_ZGERC zgerc_

  extern int M_ZAXPY(const int *n, const double complex *alpha, const double complex *x, const int *incx,
                     double complex *y, const int *incy);
  extern int M_ZGERC(const int *m, const int *n, const double complex *alpha,
                    const double complex *x, const int *incx, const double complex *y, const int *incy, 
                    double complex *a, const int *lda);
  int m,n,incx,incy,lda;
  m=n=lda=2*srOptSize;
  incx=incy=1;

//   OO[i][j] += w*O[i]*O[j] 
  M_ZGERC(&m, &n, &w, srOptO, &incx, srOptO, &incy, srOptOO, &lda);
//   HO[i] += w*e*O[i] 
  M_ZAXPY(&n, &we, srOptO, &incx, srOptHO, &incy);
  return;
}

void calculateOO(double complex *srOptOO, double complex *srOptHO, const double complex *srOptO,
                 const double w, const double complex e, const int srOptSize){
  int i,j;
  double complex tmp;
  #pragma omp parallel for default(shared) private(j,tmp)
  //    private(i,j,tmp,srOptOO)
#pragma loop noalias
  for(j=0;j<2*srOptSize;j++) {
    tmp                            = w * srOptO[j];
    srOptOO[0*(2*srOptSize)+j]    += tmp;      // update O
    srOptOO[1*(2*srOptSize)+j]    += 0.0;      // update 
    srOptHO[j]                    += e * tmp;  // update HO
  }
  
  #pragma omp parallel for default(shared) private(i,j,tmp)
#pragma loop noalias
  for(i=2;i<2*srOptSize;i++) {
    tmp            = w * srOptO[i];
    for(j=0;j<2*srOptSize;j++) {
      srOptOO[i*(2*srOptSize)+j] += w*(srOptO[j])*conj(srOptO[i]); // TBC
      //srOptOO[j+i*(2*srOptSize)] += w*(srOptO[j])*(srOptO[i]); // TBC
    }
  }

  return;
}

void calculateOO_real(double *srOptOO, double *srOptHO, const double *srOptO,
                 const double w, const double e, const int srOptSize) {
  double we=w*e;

  #define M_DAXPY daxpy_
  #define M_DGER dger_

  extern int M_DAXPY(const int *n, const double *alpha, const double *x, const int *incx,
                     double *y, const int *incy);
  extern int M_DGER(const int *m, const int *n, const double *alpha,
                    const double *x, const int *incx, const double *y, const int *incy, 
                    double *a, const int *lda);
  int m,n,incx,incy,lda;
  m=n=lda=srOptSize;
  incx=incy=1;

  /* OO[i][j] += w*O[i]*O[j] */
  M_DGER(&m, &n, &w, srOptO, &incx, srOptO, &incy, srOptOO, &lda);

  /* HO[i] += w*e*O[i] */
  M_DAXPY(&n, &we, srOptO, &incx, srOptHO, &incy);

  return;
}


void calculateQQQQ(double *qqqq, const double *lslq, const double w, const int nLSHam) {
  const int n=nLSHam*nLSHam*nLSHam*nLSHam;
  int rq,rp,ri,rj;
  int i,tmp;

  /* QQQQ[rq][rp][ri][rj] += w * LSLQ[rq][ri] * LSLQ[rp][rj] */
  # pragma omp parallel for default(shared) private(i,tmp,rq,rp,ri,rj)
  for(i=0;i<n;++i) {
    rj = i%nLSHam;   tmp=i/nLSHam;
    ri = tmp%nLSHam; tmp=tmp/nLSHam;
    rp = tmp%nLSHam; tmp=tmp/nLSHam;
    rq = tmp%nLSHam;

    qqqq[i] += w * lslq[rq*nLSHam+ri] * lslq[rp*nLSHam+rj];
  }

  return;
}

void calculateQCAQ(double *qcaq, const double *lslca, const double *lslq,
                   const double w, const int nLSHam, const int nCA) {
  const int n=nLSHam*nLSHam*nCA;
  int rq,rp,idx;
  int i,tmp;

  /* QCisAjsQ[rq][rp][idx] += w * LSLCisAjs[rq][idx] * LSLQ[rp][0] */
# pragma omp parallel for default(shared) private(i,tmp,idx,rp,rq)
  for(i=0;i<n;++i) {
    idx = i%nCA;     tmp = i/nCA;
    rp = tmp%nLSHam; tmp = tmp/nLSHam;
    rq = tmp%nLSHam;

    qcaq[i] += w * lslca[rq*nCA+idx] * lslq[rp*nLSHam];
  }

  return;
}

void calculateQCACAQ(double *qcacaq, const double *lslca, const double w,
                     const int nLSHam, const int nCA, const int nCACA,
                     int **cacaIdx) {
  const int n=nLSHam*nLSHam*nCACA;
  int rq,rp,ri,rj,idx;
  int i,tmp;

  /* QCisAjsCktAltQ[rq][rp][idx] += w * LSLCisAjs[rq][ri] * LSLCisAjs[rp][rj] */
# pragma omp parallel for default(shared) private(i,tmp,idx,rp,rq,ri,rj)
  for(i=0;i<n;++i) {
    idx = i%nCACA;   tmp = i/nCACA;
    rp = tmp%nLSHam; tmp = tmp/nLSHam;
    rq = tmp%nLSHam;

    ri = cacaIdx[idx][0];
    rj = cacaIdx[idx][1];

    qcacaq[i] += w * lslca[rq*nCA+ri] * lslca[rp*nCA+rj];
  }

  return;
}

