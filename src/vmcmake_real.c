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

void VMCMakeSample_real(MPI_Comm comm);
int makeInitialSample_real(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, /* modified by YN */ 
                      int *hiddenCfg1, int *hiddenCfg2, double complex *thetaHidden1, double complex *thetaHidden2, /* modified by YN */
                      const int qpStart, const int qpEnd, MPI_Comm comm);

void VMCMakeSample_real(MPI_Comm comm) {
  int outStep,nOutStep;
  int inStep,nInStep;
  UpdateType updateType;
  int mi,mj,ri,rj,s,t,i,hi,offset; /* modified by YN */
  int nAccept=0;
  int nFail=0; /* added by YN */
  int sample;

  double  logIpOld,logIpNew; /* logarithm of inner product <phi|L|x> */ // is this ok ? TBC
  int projCntNew[NProj];
  double         pfMNew_real[NQPFull];
  /* added by YN */
  int nVMCSampleHidden = NVMCSampleHidden; 
  int nNeuronSample = NNeuronSample;       
  int nSizeTheta = NSizeTheta; 
  int tmpHiddenCfg1[NSizeHiddenCfgSave];
  int tmpHiddenCfg2[NSizeHiddenCfgSave]; 
  double complex tmpThetaHidden1[NSizeThetaSave];
  double complex tmpThetaHidden2[NSizeThetaSave];
  double complex thetaHiddenNew1[NSizeTheta]; /* modified by KI */
  double complex thetaHiddenNew2[NSizeTheta]; /* modified by KI */
  /* added by YN */
  double x,y1,y2,w; // TBC x and y will be complex number   /* modified by YN */

  int qpStart,qpEnd;
  int rejectFlag;
  int rank,size;
  MPI_Comm_size(comm,&size);
  MPI_Comm_rank(comm,&rank);

  SplitLoop(&qpStart,&qpEnd,NQPFull,rank,size);

  
  StartTimer(30);
  if(BurnFlag==0) {
    makeInitialSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpHiddenCfg1,TmpHiddenCfg2, /* modified by YN */
                      TmpThetaHidden1,TmpThetaHidden2,qpStart,qpEnd,comm); /* modified by YN */
  } else {
    copyFromBurnSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpHiddenCfg1,TmpHiddenCfg2); /* modified by YN */
    CalcThetaHidden(TmpThetaHidden1,TmpEleNum,TmpHiddenCfg1); /* added by YN */
    CalcThetaHidden(TmpThetaHidden2,TmpEleNum,TmpHiddenCfg2); /* added by YN */
  }
  
  CalculateMAll_real(TmpEleIdx,qpStart,qpEnd);
 // printf("DEBUG: maker1: PfM=%lf\n",creal(PfM[0]));
  logIpOld = CalculateLogIP_real(PfM_real,qpStart,qpEnd,comm);
  if( !isfinite(logIpOld) ) {
    if(rank==0) fprintf(stderr,"waring: VMCMakeSample remakeSample logIpOld=%e\n",creal(logIpOld)); //TBC
    makeInitialSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpHiddenCfg1,TmpHiddenCfg2, /* modified by YN */
                      TmpThetaHidden1,TmpThetaHidden2,qpStart,qpEnd,comm); /* modified by YN */
    CalculateMAll_real(TmpEleIdx,qpStart,qpEnd);
    //printf("DEBUG: maker2: PfM=%lf\n",creal(PfM[0]));
    logIpOld = CalculateLogIP_real(PfM_real,qpStart,qpEnd,comm);
    BurnFlag = 0;
  }
  StopTimer(30);

  nOutStep = (BurnFlag==0) ? NVMCWarmUp+NVMCSample : NVMCSample+1;
  nInStep = NVMCInterval * Nsite;

  for(i=0;i<6;i++) Counter[i]=0;  /* reset counter */

  for(outStep=0;outStep<nOutStep;outStep++) {
    for(inStep=0;inStep<nInStep;inStep++) {

      for(i=0;i<nNeuronSample/10;i++) UpdateHiddenCfg(TmpHiddenCfg1,thetaHiddenNew1,TmpThetaHidden1); /* added by YN */
      for(i=0;i<nNeuronSample/10;i++) UpdateHiddenCfg(TmpHiddenCfg2,thetaHiddenNew2,TmpThetaHidden2); /* added by YN */
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
        UpdateThetaHidden(ri,rj,s,thetaHiddenNew1,TmpThetaHidden1,TmpHiddenCfg1); /* added by YN */
        UpdateThetaHidden(ri,rj,s,thetaHiddenNew2,TmpThetaHidden2,TmpHiddenCfg2); /* added by YN */
          StopTimer(60);
          StartTimer(61);
        //CalculateNewPfM2(mi,s,pfMNew,TmpEleIdx,qpStart,qpEnd);
        CalculateNewPfM2_real(mi,s,pfMNew_real,TmpEleIdx,qpStart,qpEnd);
        //printf("DEBUG: out %d in %d pfMNew=%lf \n",outStep,inStep,creal(pfMNew[0]));
          StopTimer(61);

          StartTimer(62);
        /* calculate inner product <phi|L|x> */
        //logIpNew = CalculateLogIP_fcmp(pfMNew,qpStart,qpEnd,comm);
        logIpNew = CalculateLogIP_real(pfMNew_real,qpStart,qpEnd,comm);
          StopTimer(62);

        /* Metroplis */
        x = LogProjRatio(projCntNew,TmpEleProjCnt);
        y1 = LogHiddenWeightRatio(thetaHiddenNew1,TmpThetaHidden1);  /* added by YN, modified by KI */
        y2 = LogHiddenWeightRatio(thetaHiddenNew2,TmpThetaHidden2);  /* added by YN, modified by KI */
        w = exp(2.0*(x+(logIpNew-logIpOld))+y1+y2);                   /* modified by YN */
        if( !isfinite(w) ) w = -1.0; /* should be rejected */

        if(w > genrand_real2()) { /* accept */
            // UpdateMAll will change SlaterElm, InvM (including PfM)
            StartTimer(63);
            UpdateMAll_real(mi,s,TmpEleIdx,qpStart,qpEnd);
//            UpdateMAll(mi,s,TmpEleIdx,qpStart,qpEnd);
            StopTimer(63);

          for(i=0;i<NProj;i++) TmpEleProjCnt[i] = projCntNew[i];
          for(i=0;i<NSizeTheta;i++) TmpThetaHidden1[i] = thetaHiddenNew1[i]; /* added by YN */
          for(i=0;i<NSizeTheta;i++) TmpThetaHidden2[i] = thetaHiddenNew2[i]; /* added by YN */
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
        UpdateThetaHidden(ri,rj,s,thetaHiddenNew1,TmpThetaHidden1,TmpHiddenCfg1); /* added by YN */
        UpdateThetaHidden(ri,rj,s,thetaHiddenNew2,TmpThetaHidden2,TmpHiddenCfg2); /* added by YN */
        /* The mj-th electron with spin t hops to ri */
        updateEleConfig(mj,rj,ri,t,TmpEleIdx,TmpEleCfg,TmpEleNum);
        UpdateProjCnt(rj,ri,t,projCntNew,projCntNew,TmpEleNum);
        UpdateThetaHidden(rj,ri,t,thetaHiddenNew1,thetaHiddenNew1,TmpHiddenCfg1); /* added by YN */
        UpdateThetaHidden(rj,ri,t,thetaHiddenNew2,thetaHiddenNew2,TmpHiddenCfg2); /* added by YN */

        StopTimer(65);
        StartTimer(66);

        CalculateNewPfMTwo2_real(mi, s, mj, t, pfMNew_real, TmpEleIdx, qpStart, qpEnd);
        StopTimer(66);
        StartTimer(67);

        /* calculate inner product <phi|L|x> */
        logIpNew = CalculateLogIP_real(pfMNew_real,qpStart,qpEnd,comm);

        StopTimer(67);

        /* Metroplis */
        x = LogProjRatio(projCntNew,TmpEleProjCnt);
        y1 = LogHiddenWeightRatio(thetaHiddenNew1,TmpThetaHidden1);  /* added by YN, modified by KI */
        y2 = LogHiddenWeightRatio(thetaHiddenNew2,TmpThetaHidden2);  /* added by YN, modified by KI */
        w = exp(2.0*(x+(logIpNew-logIpOld))+y1+y2); //TBC             /* modified by YN */
        if( !isfinite(w) ) w = -1.0; /* should be rejected */

        if(w > genrand_real2()) { /* accept */
          StartTimer(68);
          UpdateMAllTwo_real(mi, s, mj, t, ri, rj, TmpEleIdx,qpStart,qpEnd);
          StopTimer(68);

          for(i=0;i<NProj;i++) TmpEleProjCnt[i] = projCntNew[i];
          for(i=0;i<NSizeTheta;i++) TmpThetaHidden1[i] = thetaHiddenNew1[i]; /* added by YN */
          for(i=0;i<NSizeTheta;i++) TmpThetaHidden2[i] = thetaHiddenNew2[i]; /* added by YN */
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
        CalculateMAll_real(TmpEleIdx,qpStart,qpEnd);
        //printf("DEBUG: maker3: PfM=%lf\n",creal(PfM[0]));
        logIpOld = CalculateLogIP_real(PfM_real,qpStart,qpEnd,comm);
        /* added by YN */
        CalcThetaHidden(thetaHiddenNew1,TmpEleNum,TmpHiddenCfg1); 
        CalcThetaHidden(thetaHiddenNew2,TmpEleNum,TmpHiddenCfg2); 
        for(i=0;i<NSizeTheta;i++) { 
          if( cabs(TmpThetaHidden1[i]-thetaHiddenNew1[i]) > 1.0e-5 ) {
            fprintf(stderr,"Warning: failed in updating ThetaHidden1, %lf %lf \n",
                    cabs(TmpThetaHidden1[i]),cabs(thetaHiddenNew1[i]));
            nFail++;
            if( nFail > 20 ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
          } else if( cabs(TmpThetaHidden2[i]-thetaHiddenNew2[i]) > 1.0e-5 ) {
            fprintf(stderr,"Warning: failed in updating ThetaHidden2, %lf %lf \n",
                    cabs(TmpThetaHidden2[i]),cabs(thetaHiddenNew2[i]));
            nFail++;
            if( nFail > 20 ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
          }  
          TmpThetaHidden1[i] = thetaHiddenNew1[i];
          TmpThetaHidden2[i] = thetaHiddenNew2[i];
        } 
        /* added by YN */
        StopTimer(34);
        nAccept=0;
      }
    } /* end of instep */

    /* added by YN */
    StartTimer(73);
    for(inStep=0;inStep<nVMCSampleHidden;inStep++) {
      for(i=0;i<nNeuronSample;i++) UpdateHiddenCfg(TmpHiddenCfg1,thetaHiddenNew1,TmpThetaHidden1);
      for(i=0;i<nNeuronSample;i++) UpdateHiddenCfg(TmpHiddenCfg2,thetaHiddenNew2,TmpThetaHidden2);
      offset = inStep*nNeuronSample; 
      for(hi=0;hi<nNeuronSample;hi++){
        tmpHiddenCfg1[offset+hi] = TmpHiddenCfg1[hi];
        tmpHiddenCfg2[offset+hi] = TmpHiddenCfg2[hi];
      }
      offset = inStep*nSizeTheta; 
      for(i=0;i<nSizeTheta;i++){
        tmpThetaHidden1[offset+i] = TmpThetaHidden1[i];
        tmpThetaHidden2[offset+i] = TmpThetaHidden2[i];
      }
    }
    StopTimer(73);
    /* added by YN */

    StartTimer(35);
    /* save Electron Configuration */
    if(outStep >= nOutStep-NVMCSample) {
      sample = outStep-(nOutStep-NVMCSample);
      saveEleConfig(sample,logIpOld,TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt, /* modified by YN */
                    tmpHiddenCfg1,tmpHiddenCfg2,tmpThetaHidden1,tmpThetaHidden2); /* modified by YN */
    }
    StopTimer(35);

  } /* end of outstep */

printf("accept %lf",(double)Counter[5] / (double)Counter[4]);
  copyToBurnSample(TmpEleIdx,TmpEleCfg,TmpEleNum,TmpEleProjCnt,TmpHiddenCfg1,TmpHiddenCfg2); /* modified by YN */
  BurnFlag=1;
  return;
}

int makeInitialSample_real(int *eleIdx, int *eleCfg, int *eleNum, int *eleProjCnt, /* modified by YN */ 
                      int *hiddenCfg1, int *hiddenCfg2, double complex *thetaHidden1, double complex *thetaHidden2, /* modified by YN */
                      const int qpStart, const int qpEnd, MPI_Comm comm) {
  const int nsize = Nsize;
  const int nsite2 = Nsite2;
  const int nNeuronSample = NNeuronSample; /* added by YN */
  int flag=1,flagRdc,loop=0;
  int ri,mi,si,msi,rsi,hi; /* modified by YN */
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

    /* added by YN */
    for(hi=0;hi<nNeuronSample;hi++) hiddenCfg1[hi] = (genrand_real2()<0.5) ? 1 : -1;
    for(hi=0;hi<nNeuronSample;hi++) hiddenCfg2[hi] = (genrand_real2()<0.5) ? 1 : -1;
    CalcThetaHidden(thetaHidden1,eleNum,hiddenCfg1); 
    CalcThetaHidden(thetaHidden2,eleNum,hiddenCfg2); 
    /* added by YN */

    flag = CalculateMAll_real(eleIdx,qpStart,qpEnd);
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
