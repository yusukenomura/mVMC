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
 * make sample
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

void VMCMakeSample(MPI_Comm comm);
/* modified by YN, modified by KI */ 
int makeInitialSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double complex *thetaHidden,
                      const int qpStart, const int qpEnd, MPI_Comm comm);
void copyFromBurnSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt);
void copyToBurnSample(const int *eleIdx, const int *eleCfg, const int *eleNum, const int *eleProjCnt);
void saveEleConfig(const int sample, const double complex logIp,
                   const int *eleIdx, const int *eleCfg, const int *eleNum, 
                   const int *eleProjCnt, const double complex *thetaHidden );
/* modified by YN, modified by KI */ 
void sortEleConfig(int *eleIdx, int *eleCfg, const int *eleNum);
void ReduceCounter(MPI_Comm comm);
void makeCandidate_hopping(int *mi_, int *ri_, int *rj_, int *s_, int *rejectFlag_,
                           const int *eleIdx, const int *eleCfg);
void makeCandidate_exchange(int *mi_, int *ri_, int *rj_, int *s_, int *rejectFlag_,
                            const int *eleIdx, const int *eleCfg, const int *eleNum);
void updateEleConfig(int mi, int ri, int rj, int s,
                     int *eleIdx, int *eleCfg, int *eleNum);
void revertEleConfig(int mi, int ri, int rj, int s,
                     int *eleIdx, int *eleCfg, int *eleNum);

typedef enum {HOPPING, EXCHANGE, NONE} UpdateType;
UpdateType getUpdateType(int path);

void VMCMakeSample(MPI_Comm comm) {
  int outStep,nOutStep;
  int inStep,nInStep;
  UpdateType updateType;
  int mi,mj,ri,rj,s,t,i;
  int nAccept=0;
  int nFail=0; /* added by YN */
  int sample;

  double complex logIpOld,logIpNew; /* logarithm of inner product <phi|L|x> */ // is this ok ? TBC
  int projCntNew[NProj];
  double complex pfMNew[NQPFull];
  double complex thetaHiddenNew[NSizeTheta]; /* added by YN, modified by KI*/
  double x,y,w; // TBC x and y will be complex number   /* modified by YN */

  int qpStart,qpEnd;
  int rejectFlag;
  int rank,size;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  SplitLoop(&qpStart,&qpEnd,NQPFull,rank,size);
  /* added by YN */
    fprintf(stderr, " make not implemented .\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  /* added by YN */

  StartTimer(30);
  if(BurnFlag==0) {
    makeInitialSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpThetaHidden, /* modified by YN */
                      qpStart,qpEnd,comm);
  } else {
    copyFromBurnSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt);
    CalcThetaHidden(TmpThetaHidden,TmpEleNum); /* added by YN */
  }
  
  CalculateMAll_fcmp(TmpEleIdx,qpStart,qpEnd);
 // printf("DEBUG: maker1: PfM=%lf\n",creal(PfM[0]));
  logIpOld = CalculateLogIP_fcmp(PfM,qpStart,qpEnd,comm);
  if( !isfinite(creal(logIpOld) + cimag(logIpOld)) ) {
    if(rank==0) fprintf(stderr,"waring: VMCMakeSample remakeSample logIpOld=%e\n",creal(logIpOld)); //TBC
    makeInitialSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpThetaHidden, /* modified by YN */
                      qpStart,qpEnd,comm);
    CalculateMAll_fcmp(TmpEleIdx,qpStart,qpEnd);
    //printf("DEBUG: maker2: PfM=%lf\n",creal(PfM[0]));
    logIpOld = CalculateLogIP_fcmp(PfM,qpStart,qpEnd,comm);
    BurnFlag = 0;
  }
  StopTimer(30);

  nOutStep = (BurnFlag==0) ? NVMCWarmUp+NVMCSample : NVMCSample+1;
  nInStep = NVMCInterval * Nsite;

  for(i=0;i<4;i++) Counter[i]=0;  /* reset counter */

  for(outStep=0;outStep<nOutStep;outStep++) {
    for(inStep=0;inStep<nInStep;inStep++) {

      updateType = getUpdateType(NExUpdatePath);

      if(updateType==HOPPING) { /* hopping */
        Counter[0]++;

        StartTimer(31);
        makeCandidate_hopping(&mi, &ri, &rj, &s, &rejectFlag,
                              TmpEleIdx, TmpEleCfg);
        StopTimer(31);

        if(rejectFlag) continue;

        StartTimer(32);
          StartTimer(60);
        /* The mi-th electron with spin s hops to site rj */
        updateEleConfig(mi,ri,rj,s,TmpEleIdx,TmpEleCfg,TmpEleNum);
        UpdateProjCnt(ri,rj,s,projCntNew,TmpEleProjCnt,TmpEleNum);
        UpdateThetaHidden(ri,rj,s,thetaHiddenNew,TmpThetaHidden); /* added by YN */
          StopTimer(60);
          StartTimer(61);
        //CalculateNewPfM2(mi,s,pfMNew,TmpEleIdx,qpStart,qpEnd);
        CalculateNewPfM2(mi,s,pfMNew,TmpEleIdx,qpStart,qpEnd);
        //printf("DEBUG: out %d in %d pfMNew=%lf \n",outStep,inStep,creal(pfMNew[0]));
          StopTimer(61);

          StartTimer(62);
        /* calculate inner product <phi|L|x> */
        //logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);
        logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);
          StopTimer(62);

        /* Metroplis */
        x = LogProjRatio(projCntNew,TmpEleProjCnt);
        y = LogHiddenWeightRatio(thetaHiddenNew,TmpThetaHidden);  /* added by YN, modified by KI */
        w = exp(2.0*(x+y+creal(logIpNew-logIpOld)));              /* modified by YN */
        if( !isfinite(w) ) w = -1.0; /* should be rejected */

        if(w > genrand_real2()) { /* accept */
            // UpdateMAll will change SlaterElm, InvM (including PfM)
            StartTimer(63);
            UpdateMAll(mi,s,TmpEleIdx,qpStart,qpEnd);
//            UpdateMAll(mi,s,TmpEleIdx,qpStart,qpEnd);
            StopTimer(63);

          for(i=0;i<NProj;i++) TmpEleProjCnt[i] = projCntNew[i];
          for(i=0;i<NSizeTheta;i++) TmpThetaHidden[i] = thetaHiddenNew[i]; /* added by YN */
          logIpOld = logIpNew;
          nAccept++;
          Counter[1]++;
        } else { /* reject */
          revertEleConfig(mi,ri,rj,s,TmpEleIdx,TmpEleCfg,TmpEleNum);
        }
        StopTimer(32);

      } else if(updateType==EXCHANGE) { /* exchange */
        Counter[2]++;

        StartTimer(31);
        makeCandidate_exchange(&mi, &ri, &rj, &s, &rejectFlag,
                               TmpEleIdx, TmpEleCfg, TmpEleNum);
        StopTimer(31);

        if(rejectFlag) continue;

        StartTimer(33);
        StartTimer(65);

        /* The mi-th electron with spin s exchanges with the electron on site rj with spin 1-s */
        t = 1-s;
        mj = TmpEleCfg[rj+t*Nsite];

        /* The mi-th electron with spin s hops to rj */
        updateEleConfig(mi,ri,rj,s,TmpEleIdx,TmpEleCfg,TmpEleNum);
        UpdateProjCnt(ri,rj,s,projCntNew,TmpEleProjCnt,TmpEleNum);
        UpdateThetaHidden(ri,rj,s,thetaHiddenNew,TmpThetaHidden); /* added by YN */
        /* The mj-th electron with spin t hops to ri */
        updateEleConfig(mj,rj,ri,t,TmpEleIdx,TmpEleCfg,TmpEleNum);
        UpdateProjCnt(rj,ri,t,projCntNew,projCntNew,TmpEleNum);
        UpdateThetaHidden(rj,ri,t,thetaHiddenNew,thetaHiddenNew); /* added by YN */

        StopTimer(65);
        StartTimer(66);

        CalculateNewPfMTwo2_fcmp(mi, s, mj, t, pfMNew, TmpEleIdx, qpStart, qpEnd);
        StopTimer(66);
        StartTimer(67);

        /* calculate inner product <phi|L|x> */
        logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);

        StopTimer(67);

        /* Metroplis */
        x = LogProjRatio(projCntNew,TmpEleProjCnt);
        y = LogHiddenWeightRatio(thetaHiddenNew,TmpThetaHidden);  /* added by YN, modified by KI */
        w = exp(2.0*(x+y+creal(logIpNew-logIpOld))); //TBC        /* modified by YN */
        if( !isfinite(w) ) w = -1.0; /* should be rejected */

        if(w > genrand_real2()) { /* accept */
          StartTimer(68);
          UpdateMAllTwo_fcmp(mi, s, mj, t, ri, rj, TmpEleIdx,qpStart,qpEnd);
          StopTimer(68);

          for(i=0;i<NProj;i++) TmpEleProjCnt[i] = projCntNew[i];
          for(i=0;i<NSizeTheta;i++) TmpThetaHidden[i] = thetaHiddenNew[i]; /* added by YN */
          logIpOld = logIpNew;
          nAccept++;
          Counter[3]++;
        } else { /* reject */
          revertEleConfig(mj,rj,ri,t,TmpEleIdx,TmpEleCfg,TmpEleNum);
          revertEleConfig(mi,ri,rj,s,TmpEleIdx,TmpEleCfg,TmpEleNum);
        }
        StopTimer(33);
      }

      if(nAccept>NPfUpdate) { /* modified by YN */
        StartTimer(34);
        /* recal PfM and InvM */
        CalculateMAll_fcmp(TmpEleIdx,qpStart,qpEnd);
        //printf("DEBUG: maker3: PfM=%lf\n",creal(PfM[0]));
        logIpOld = CalculateLogIP_fcmp(PfM,qpStart,qpEnd,comm);
        /* added by YN, modified by KI */
        CalcThetaHidden(thetaHiddenNew,TmpEleNum); 
        for(i=0;i<NSizeTheta;i++) { 
          if( cabs(TmpThetaHidden[i]-thetaHiddenNew[i]) > 1.0e-5 ) {
            fprintf(stderr,"Warning: failed in updating ThetaHidden, %lf %lf \n",
                    cabs(TmpThetaHidden[i]),cabs(thetaHiddenNew[i]));
            nFail++;
            if( nFail > 20 ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
          }  
          TmpThetaHidden[i] = thetaHiddenNew[i];
        } 
        /* added by YN */
        StopTimer(34);
        nAccept=0;
      }
    } /* end of instep */

    StartTimer(35);
    /* save Electron Configuration */
    if(outStep >= nOutStep-NVMCSample) {
      sample = outStep-(nOutStep-NVMCSample);
      saveEleConfig(sample,logIpOld,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpThetaHidden); /* modified by YN */
    }
    StopTimer(35);

  } /* end of outstep */

  copyToBurnSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt);
  BurnFlag=1;
  return;
}

int makeInitialSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, double complex *thetaHidden, /* modified by YN, modified by KI */
                      const int qpStart, const int qpEnd, MPI_Comm comm) {
  const int nsize = Nsize;
  const int nsite2 = Nsite2;
  int flag=1,flagRdc,loop=0;
  int ri,mi,si,msi,rsi;
  int rank,size;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);
  
  do {
    /* initialize */
    #pragma omp parallel for default(shared) private(msi)
    for(msi=0;msi<nsize;msi++) eleIdx[msi] = -1;
    #pragma omp parallel for default(shared) private(rsi)
    for(rsi=0;rsi<nsite2;rsi++) eleCfg[rsi] = -1;
    
    /* local spin */
    for(ri=0;ri<Nsite;ri++) {
      if(LocSpn[ri]==1) {
        do {
          mi = gen_rand32()%Ne;
          si = (genrand_real2()<0.5) ? 0 : 1;
        } while(eleIdx[mi+si*Ne]!=-1);
        eleCfg[ri+si*Nsite] = mi;
        eleIdx[mi+si*Ne] = ri;
      }
    }
    
    /* itinerant electron */
    for(si=0;si<2;si++) {
      for(mi=0;mi<Ne;mi++) {
        if(eleIdx[mi+si*Ne]== -1) {
          do {
            ri = gen_rand32()%Nsite;
          } while (eleCfg[ri+si*Nsite]!= -1 || LocSpn[ri]==1);
          eleCfg[ri+si*Nsite] = mi;
          eleIdx[mi+si*Ne] = ri;
        }
      }
    }
    
    /* EleNum */
    #pragma omp parallel for default(shared) private(rsi)
    #pragma loop noalias
    for(rsi=0;rsi<nsite2;rsi++) {
      eleNum[rsi] = (eleCfg[rsi] < 0) ? 0 : 1;
    }
    
    MakeProjCnt(eleProjCnt,eleNum);
    CalcThetaHidden(thetaHidden,eleNum); /* added by YN */

    flag = CalculateMAll_fcmp(eleIdx,qpStart,qpEnd);
    //printf("DEBUG: maker4: PfM=%lf\n",creal(PfM[0]));
    if(size>1) {
      MPI_Allreduce(&flag,&flagRdc,1,MPI_INT,MPI_MAX,comm);
      flag = flagRdc;
    }

    loop++;
    if(loop>100) {
      if(rank==0) fprintf(stderr, "error: makeInitialSample: Too many loops\n");
      MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
    }
  } while (flag>0);
 
  return 0;
}

void copyFromBurnSample(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt) {
  int i,n;
  const int *burnEleIdx = BurnEleIdx;
  n = Nsize + 2*Nsite + 2*Nsite + NProj;
  #pragma loop noalias
  for(i=0;i<n;i++) eleIdx[i] = burnEleIdx[i];
  return;
}

void copyToBurnSample(const int *eleIdx, const int *eleCfg, const int *eleNum, const int *eleProjCnt) {
  int i,n;
  int *burnEleIdx = BurnEleIdx;
  n = Nsize + 2*Nsite + 2*Nsite + NProj;
  #pragma loop noalias
  for(i=0;i<n;i++) burnEleIdx[i] = eleIdx[i];
  return;
}

void saveEleConfig(const int sample, const double complex logIp,
                   const int *eleIdx, const int *eleCfg, const int *eleNum, /* modified by YN */
                   const int *eleProjCnt, const double complex *thetaHidden ) {     /* modified by YN, modified by KI */ 
  int i,offset;
  double x,y;  /* modified by YN */
  const int nsize=Nsize;
  const int nsite2 = Nsite2;
  const int nProj = NProj;
  const int nSizeTheta = NSizeTheta; /* added by YN */

  offset = sample*nsize;
  #pragma loop noalias
  for(i=0;i<nsize;i++) EleIdx[offset+i] = eleIdx[i];
  offset = sample*nsite2;
  #pragma loop noalias
  for(i=0;i<nsite2;i++) EleCfg[offset+i] = eleCfg[i];
  #pragma loop noalias
  for(i=0;i<nsite2;i++) EleNum[offset+i] = eleNum[i];
  offset = sample*nProj;
  #pragma loop noalias
  for(i=0;i<nProj;i++) EleProjCnt[offset+i] = eleProjCnt[i];
  /* added by YN */
  offset = sample*nSizeTheta;
  #pragma loop noalias
  for(i=0;i<nSizeTheta;i++) ThetaHidden[offset+i] = thetaHidden[i];
  /* added by YN */
  
  x = LogProjVal(eleProjCnt);
  y = LogHiddenWeightVal(thetaHidden); /* added by YN, modified by KI */
  logSqPfFullSlater[sample] = 2.0*(x+y+creal(logIp));//TBC /* modified by YN */
  
  return;
}

void sortEleConfig(int *eleIdx, int *eleCfg, const int *eleNum) {
/*   int ri,mi=0; */
/*   for(ri=0;ri<Nsite;ri++) { */
/*     if(eleNum[ri]>0) { */
/*       eleCfg[ri]=mi; */
/*       eleIdx[mi]=ri; */
/*       mi++; */
/*     } else { */
/*       eleCfg[ri]=-1; */
/*     } */
/*   } */
/*   mi=0; */
/*   for(ri=0;ri<Nsite;ri++) { */
/*     if(eleNum[ri+Nsite]>0) { */
/*       eleCfg[ri+Nsite]=mi; */
/*       eleIdx[mi+Ne]=ri; */
/*       mi++; */
/*     } else { */
/*       eleCfg[ri+Nsite]=-1; */
/*     } */
/*   } */

  return;
}

void ReduceCounter(MPI_Comm comm) {
  #ifdef _mpi_use
  int n=4;
  int recv[n];
  int i;
  int rank,size;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  MPI_Allreduce(Counter,recv,n,MPI_INT,MPI_SUM,comm);
  if(rank==0) {
    for(i=0;i<n;i++) Counter[i] = recv[i];
  }
  #endif
  return;
}


/* The mi-th electron with spin s hops to site rj */
void makeCandidate_hopping(int *mi_, int *ri_, int *rj_, int *s_, int *rejectFlag_,
                           const int *eleIdx, const int *eleCfg) {
  const int icnt_max = Nsite*Nsite;
  int icnt;
  int mi, ri, rj, s, flag;

  flag = 0; // FALSE
  do {
    mi = gen_rand32()%Ne;
    s = (genrand_real2()<0.5) ? 0 : 1;
    ri = eleIdx[mi+s*Ne];
  } while (LocSpn[ri] == 1);

  icnt = 0;
  do {
    rj = gen_rand32()%Nsite;
    if(icnt> icnt_max){
      flag = 1; // TRUE
      break;
    }
    icnt+=1;
  } while (eleCfg[rj+s*Nsite] != -1 || LocSpn[rj]==1);

  *mi_ = mi;
  *ri_ = ri;
  *rj_ = rj;
  *s_ = s;
  *rejectFlag_ = flag;

  return;
}

/* The mi-th electron with spin s exchanges with the electron on site rj with spin 1-s */
void makeCandidate_exchange(int *mi_, int *ri_, int *rj_, int *s_, int *rejectFlag_,
                           const int *eleIdx, const int *eleCfg, const int *eleNum) {
  int mi, mj, ri, rj, s, t, flag;

  flag = 1; // TRUE
  for(ri=0;ri<Nsite;ri++){
    if((eleNum[ri]+eleNum[ri+Nsite]) == 1){
      flag = 0; // FALSE
      break;
    }
  }
  if(flag) {
    *rejectFlag_ = flag;
    return;
  }

  do {
    mi = gen_rand32()%Ne;
    s = (genrand_real2()<0.5) ? 0 : 1;
    ri = eleIdx[mi+s*Ne];
  } while (eleCfg[ri+(1-s)*Nsite] != -1);
  do {
    mj = gen_rand32()%Ne;
    t = 1-s;
    rj = eleIdx[mj+t*Ne];
  } while (eleCfg[rj+(1-t)*Nsite] != -1);

  *mi_ = mi;
  *ri_ = ri;
  *rj_ = rj;
  *s_ = s;
  *rejectFlag_ = flag;

  return;
}

/* The mi-th electron with spin s hops to site rj */
void updateEleConfig(int mi, int ri, int rj, int s,
                     int *eleIdx, int *eleCfg, int *eleNum) {
  eleIdx[mi+s*Ne] = rj;
  eleCfg[ri+s*Nsite] = -1;
  eleCfg[rj+s*Nsite] = mi;
  eleNum[ri+s*Nsite] = 0;
  eleNum[rj+s*Nsite] = 1;
  return;
}

void revertEleConfig(int mi, int ri, int rj, int s,
                     int *eleIdx, int *eleCfg, int *eleNum) {
  eleIdx[mi+s*Ne] = ri;
  eleCfg[ri+s*Nsite] = mi;
  eleCfg[rj+s*Nsite] = -1;
  eleNum[ri+s*Nsite] = 1;
  eleNum[rj+s*Nsite] = 0;
  return;
}


UpdateType getUpdateType(int path) {
  if(path==0) {
    return HOPPING;
  } else if (path==1) {
    return (genrand_real2()<0.001) ? EXCHANGE : HOPPING; /* exchange or hopping */
  } else if (path==2) {
    return EXCHANGE;
  }
  return NONE;
}
