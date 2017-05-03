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
 * calculate Hamiltonian
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

/* modified by YN */
void CalculateHamiltonian(double complex *e, const double complex ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                          const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                          const double complex *thetaHidden1, const double complex *thetaHidden2);
double complex CalculateHamiltonian0(const int *eleNum);
void CalculateHamiltonian1(double complex *e, const double complex ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                           const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                           const double complex *thetaHidden1, const double complex *thetaHidden2);
void CalculateHamiltonian2(double complex *e, const double complex ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                           const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                           const double complex *thetaHidden1, const double complex *thetaHidden2);

void CalculateHamiltonian(double complex *e, const double complex ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                          const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                          const double complex *thetaHidden1, const double complex *thetaHidden2){
/* modified by YN */
  const int *n0 = eleNum;
  const int *n1 = eleNum + Nsite;
  double complex *myTmp; /* modified by YN */
  int idx;
  int ri,rj,s,rk,rl,t;
  int *myEleIdx, *myEleNum, *myProjCntNew;
  /* added by YN */
  const int nVMCSampleHidden2 = 2*NVMCSampleHidden;
  int i;
  int *myHiddenCfgNew1;
  int *myHiddenCfgNew2;
  double complex *myThetaHiddenNew1; 
  double complex *myThetaHiddenNew2; 
  /* added by YN */
  double complex *myBuffer;
  double complex *myEnergy, myEnergy_tmp; /* modified by YN */

  RequestWorkSpaceThreadInt(Nsize+Nsite2+NProj+2*NNeuronSample); /* modified by YN */
  RequestWorkSpaceThreadComplex(2*NSizeTheta+4*NVMCSampleHidden+NQPFull+2*Nsize); /* modified by KI  */
  /* GreenFunc1: NQPFull, GreenFunc2: NQPFull+2*Nsize */
  for(i=0;i<2*NVMCSampleHidden;i++) e[i] = 0.0; /* added by YN */

  /*
#pragma omp parallel default(shared)\
  private(myEleIdx,myEleNum,myProjCntNew,myBuffer,myEnergy,idx)  \
  reduction(+:e)
  */
/* modified by YN */
#pragma omp parallel default(none) \
  private(myEleIdx, myEleNum, myProjCntNew, myBuffer, myEnergy, myEnergy_tmp, myTmp, idx, i, ri, rj, rk, rl, s, t, \
          myHiddenCfgNew1, myHiddenCfgNew2, myThetaHiddenNew1, myThetaHiddenNew2) \
  firstprivate(ip, Nsize, Nsite2, NProj, NSizeTheta, NNeuronSample, NQPFull, NCoulombIntra, CoulombIntra, ParaCoulombIntra, \
               NCoulombInter, CoulombInter, ParaCoulombInter, NHundCoupling, HundCoupling, ParaHundCoupling, \
               NTransfer, Transfer, ParaTransfer, NPairHopping, PairHopping, ParaPairHopping, nVMCSampleHidden2, \
               NExchangeCoupling, ExchangeCoupling, ParaExchangeCoupling, NInterAll, InterAll, ParaInterAll, n0, n1) \
  shared(eleCfg, eleProjCnt, hiddenCfg1, hiddenCfg2, thetaHidden1, thetaHidden2, eleIdx, eleNum, e) 
/* modified by YN */
  {
    myEleIdx = GetWorkSpaceThreadInt(Nsize);
    myEleNum = GetWorkSpaceThreadInt(Nsite2);
    myProjCntNew = GetWorkSpaceThreadInt(NProj);
    /* added by YN */
    myHiddenCfgNew1 = GetWorkSpaceThreadInt(NNeuronSample); 
    myHiddenCfgNew2 = GetWorkSpaceThreadInt(NNeuronSample); 
    myThetaHiddenNew1 = GetWorkSpaceThreadComplex(NSizeTheta); /* modified by KI */
    myThetaHiddenNew2 = GetWorkSpaceThreadComplex(NSizeTheta); /* modified by KI */
    myEnergy = GetWorkSpaceThreadComplex(nVMCSampleHidden2);
    myTmp    = GetWorkSpaceThreadComplex(nVMCSampleHidden2);
    /* added by YN */
    myBuffer = GetWorkSpaceThreadComplex(NQPFull+2*Nsize);

    #pragma loop noalias
    for(idx=0;idx<Nsize;idx++) myEleIdx[idx] = eleIdx[idx];
    #pragma loop noalias
    for(idx=0;idx<Nsite2;idx++) myEleNum[idx] = eleNum[idx];
    #pragma omp barrier
    
    /* modified by YN */
    #pragma loop noalias
    for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] = 0.0;
    myEnergy_tmp = 0.0;
    /* modified by YN */

    #pragma omp master
    {StartTimer(70);}

    /* CoulombIntra */
    #pragma omp for private(idx,ri) nowait
    for(idx=0;idx<NCoulombIntra;idx++) {
      ri = CoulombIntra[idx];
      myEnergy_tmp += ParaCoulombIntra[idx] * n0[ri] * n1[ri]; /* modified by YN */
    }
  
    /* CoulombInter */
    #pragma omp for private(idx,ri,rj) nowait
    for(idx=0;idx<NCoulombInter;idx++) {
      ri = CoulombInter[idx][0];
      rj = CoulombInter[idx][1];
      myEnergy_tmp += ParaCoulombInter[idx] * (n0[ri]+n1[ri]) * (n0[rj]+n1[rj]); /* modified by YN */
    }

    /* HundCoupling */
    #pragma omp for private(idx,ri,rj) nowait
    for(idx=0;idx<NHundCoupling;idx++) {
      ri = HundCoupling[idx][0];
      rj = HundCoupling[idx][1];
      myEnergy_tmp -= ParaHundCoupling[idx] * (n0[ri]*n0[rj] + n1[ri]*n1[rj]); /* modified by YN */
      /* Caution: negative sign */
    }

    #pragma omp master
    {StopTimer(70);StartTimer(71);}

    /* Transfer */
    #pragma omp for private(idx,ri,rj,s) schedule(dynamic) nowait
    for(idx=0;idx<NTransfer;idx++) {
      ri = Transfer[idx][0];
      rj = Transfer[idx][2];
      s  = Transfer[idx][3];
      
      /* modified by YN */
      GreenFunc1(myTmp,ri,rj,s,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew, 
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] -= ParaTransfer[idx] * myTmp[i];
      /* modified by YN */
      /* Caution: negative sign */
    }

    #pragma omp master
    {StopTimer(71);StartTimer(72);}

    /* Pair Hopping */
    #pragma omp for private(idx,ri,rj) schedule(dynamic) nowait
    for(idx=0;idx<NPairHopping;idx++) {
      ri = PairHopping[idx][0];
      rj = PairHopping[idx][1];
    
      /* modified by YN */
      GreenFunc2(myTmp,ri,rj,ri,rj,0,1,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaPairHopping[idx] * myTmp[i];
      /* modified by YN */
    }

    /* Exchange Coupling */
    #pragma omp for private(idx,ri,rj,myTmp) schedule(dynamic) nowait
    for(idx=0;idx<NExchangeCoupling;idx++) {
      ri = ExchangeCoupling[idx][0];
      rj = ExchangeCoupling[idx][1];
    
      /* modified by YN */
      GreenFunc2(myTmp,ri,rj,rj,ri,0,1,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaExchangeCoupling[idx] * myTmp[i];

      GreenFunc2(myTmp,ri,rj,rj,ri,1,0,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaExchangeCoupling[idx] * myTmp[i];
      /* modified by YN */
    }

    /* Inter All */
    #pragma omp for private(idx,ri,rj,s,rk,rl,t) schedule(dynamic) nowait
    for(idx=0;idx<NInterAll;idx++) {
      ri = InterAll[idx][0];
      rj = InterAll[idx][2];
      s  = InterAll[idx][3];
      rk = InterAll[idx][4];
      rl = InterAll[idx][6];
      t  = InterAll[idx][7];
      
      /* modified by YN */
      GreenFunc2(myTmp,ri,rj,rk,rl,s,t,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaInterAll[idx] * myTmp[i];
      /* modified by YN */
    }

    #pragma omp master
    {StopTimer(72);}

    /* modified by YN */
    #pragma omp critical
    {
      for(i=0;i<nVMCSampleHidden2;i++) e[i] += myEnergy[i] + myEnergy_tmp;
    }
    /* modified by YN */
  }

  ReleaseWorkSpaceThreadInt();
  ReleaseWorkSpaceThreadComplex();
  return; /* modified by YN */
}

/* Calculate the CoulombIntra, CoulombInter, Hund terms, */
/* which can be calculated by number operators. */
/* This function will be used in the Lanczos mode */
double complex CalculateHamiltonian0(const int *eleNum) {
  const int *n0 = eleNum;
  const int *n1 = eleNum + Nsite;
  double complex e=0.0;
  int idx;
  int ri,rj;
  double complex myEnergy;

#pragma omp parallel default(shared)\
  private(myEnergy) reduction(+:e)
  {
    myEnergy = 0.0;

    /* CoulombIntra */
    #pragma omp for private(idx,ri)
    for(idx=0;idx<NCoulombIntra;idx++) {
      ri = CoulombIntra[idx];
      myEnergy += ParaCoulombIntra[idx] * n0[ri] * n1[ri];
    }
  
    /* CoulombInter */
    #pragma omp for private(idx,ri,rj)
    for(idx=0;idx<NCoulombInter;idx++) {
      ri = CoulombInter[idx][0];
      rj = CoulombInter[idx][1];
      myEnergy += ParaCoulombInter[idx] * (n0[ri]+n1[ri]) * (n0[rj]+n1[rj]);
    }

    /* HundCoupling */
    #pragma omp for private(idx,ri,rj)
    for(idx=0;idx<NHundCoupling;idx++) {
      ri = HundCoupling[idx][0];
      rj = HundCoupling[idx][1];
      myEnergy -= ParaHundCoupling[idx] * (n0[ri]*n0[rj] + n1[ri]*n1[rj]);
      /* Caution: negative sign */
    }

    e += myEnergy;
  }

  return e;
}

/* Calculate the transfer terms, */
/* which can be calculated by 1-body Green function. */
/* This function will be used in the Lanczos mode */
/* modified by YN */
void CalculateHamiltonian1(double complex *e, const double complex ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                           const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                           const double complex *thetaHidden1, const double complex *thetaHidden2){
/* modified by YN */
  double complex *myTmp; /* modified by YN */
  int idx;
  int ri,rj,s;
  int *myEleIdx, *myEleNum, *myProjCntNew;
  /* added by YN */
  const int nVMCSampleHidden2 = 2*NVMCSampleHidden;
  int i;
  int *myHiddenCfgNew1;
  int *myHiddenCfgNew2;
  double complex *myThetaHiddenNew1; 
  double complex *myThetaHiddenNew2; 
  /* added by YN */
  double complex *myBuffer;
  double complex *myEnergy; /* modified by YN */

  RequestWorkSpaceThreadInt(Nsize+Nsite2+NProj+2*NNeuronSample);
  RequestWorkSpaceThreadComplex(2*NSizeTheta+4*NVMCSampleHidden+NQPFull); /* modified by KI, modified by YN */
  /* GreenFunc1: NQPFull */
  for(i=0;i<2*NVMCSampleHidden;i++) e[i] = 0.0; /* added by YN */

/* modified by YN */
#pragma omp parallel default(shared)\
  private(myEleIdx,myEleNum,myProjCntNew,myBuffer,myEnergy,myTmp,idx,i,  \
          myHiddenCfgNew1,myHiddenCfgNew2,myThetaHiddenNew1,myThetaHiddenNew2)\
/* modified by YN */
  {
    myEleIdx = GetWorkSpaceThreadInt(Nsize);
    myEleNum = GetWorkSpaceThreadInt(Nsite2);
    myProjCntNew = GetWorkSpaceThreadInt(NProj);
    /* added by YN */
    myHiddenCfgNew1 = GetWorkSpaceThreadInt(NNeuronSample);
    myHiddenCfgNew2 = GetWorkSpaceThreadInt(NNeuronSample);
    myThetaHiddenNew1 = GetWorkSpaceThreadComplex(NSizeTheta); 
    myThetaHiddenNew2 = GetWorkSpaceThreadComplex(NSizeTheta); 
    myEnergy = GetWorkSpaceThreadComplex(nVMCSampleHidden2);
    myTmp    = GetWorkSpaceThreadComplex(nVMCSampleHidden2);
    /* added by YN */
    myBuffer = GetWorkSpaceThreadComplex(NQPFull);

    #pragma loop noalias
    for(idx=0;idx<Nsize;idx++) myEleIdx[idx] = eleIdx[idx];
    #pragma loop noalias
    for(idx=0;idx<Nsite2;idx++) myEleNum[idx] = eleNum[idx];

    /* modified by YN */
    #pragma loop noalias
    for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] = 0.0;
    /* modified by YN */

    /* Transfer */
    #pragma omp for private(idx,ri,rj,s) schedule(dynamic) nowait
    for(idx=0;idx<NTransfer;idx++) {
      ri = Transfer[idx][0];
      rj = Transfer[idx][2];
      s  = Transfer[idx][3];
      
      /* modified by YN */
      GreenFunc1(myTmp,ri,rj,s,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew, 
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] -= ParaTransfer[idx] * myTmp[i];
      /* modified by YN */
      /* Caution: negative sign */
    }

    /* modified by YN */
    #pragma omp critical
    {  
      for(i=0;i<nVMCSampleHidden2;i++) e[i] += myEnergy[i];
    }
  }

  ReleaseWorkSpaceThreadInt();
  ReleaseWorkSpaceThreadComplex();
  return; /* modified by YN */
}

/* Calculate the exchange coupling, pair hopping, interAll terms, */
/* which can be calculated by 2-body Green function. */
/* This function will be used in the Lanczos mode */
/* modified by YN */
void CalculateHamiltonian2(double complex *e, const double complex ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                           const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                           const double complex *thetaHidden1, const double complex *thetaHidden2){
/* modified by YN */
  double complex *myTmp; /* modified by KI, modified by YN  */
  int idx;
  int ri,rj,s,rk,rl,t;
  int *myEleIdx, *myEleNum, *myProjCntNew;
  /* added by YN */
  const int nVMCSampleHidden2 = 2*NVMCSampleHidden;
  int i;
  int *myHiddenCfgNew1;
  int *myHiddenCfgNew2;
  double complex *myThetaHiddenNew1; 
  double complex *myThetaHiddenNew2; 
  /* added by YN */
  double complex *myBuffer;
  double complex *myEnergy; /* modified by YN */

  RequestWorkSpaceThreadInt(Nsize+Nsite2+NProj+2*NNeuronSample); /* modified by YN */
  RequestWorkSpaceThreadComplex(2*NSizeTheta+4*NVMCSampleHidden+NQPFull+2*Nsize); /* modified by KI, modified by YN */
  /* GreenFunc2: NQPFull+2*Nsize */
  for(i=0;i<2*NVMCSampleHidden;i++) e[i] = 0.0; /* added by YN */

/* modified by YN */
#pragma omp parallel default(shared)\
  private(myEleIdx,myEleNum,myProjCntNew,myBuffer,myEnergy,myTmp,idx,i,  \
          myHiddenCfgNew1,myHiddenCfgNew2,myThetaHiddenNew1,myThetaHiddenNew2) \
/* modified by YN */
  {
    myEleIdx = GetWorkSpaceThreadInt(Nsize);
    myEleNum = GetWorkSpaceThreadInt(Nsite2);
    myProjCntNew = GetWorkSpaceThreadInt(NProj);
    /* added by YN */
    myHiddenCfgNew1 = GetWorkSpaceThreadInt(NNeuronSample); 
    myHiddenCfgNew2 = GetWorkSpaceThreadInt(NNeuronSample); 
    myThetaHiddenNew1 = GetWorkSpaceThreadComplex(NSizeTheta); 
    myThetaHiddenNew2 = GetWorkSpaceThreadComplex(NSizeTheta); 
    myEnergy = GetWorkSpaceThreadComplex(nVMCSampleHidden2); 
    myTmp    = GetWorkSpaceThreadComplex(nVMCSampleHidden2); 
    /* added by YN */
    myBuffer = GetWorkSpaceThreadComplex(NQPFull+2*Nsize);

    #pragma loop noalias
    for(idx=0;idx<Nsize;idx++) myEleIdx[idx] = eleIdx[idx];
    #pragma loop noalias
    for(idx=0;idx<Nsite2;idx++) myEleNum[idx] = eleNum[idx];

    /* modified by YN */
    #pragma loop noalias
    for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] = 0.0;
    /* modified by YN */

    /* Pair Hopping */
    #pragma omp for private(idx,ri,rj) schedule(dynamic) nowait
    for(idx=0;idx<NPairHopping;idx++) {
      ri = PairHopping[idx][0];
      rj = PairHopping[idx][1];
    
      /* modified by YN */
      GreenFunc2(myTmp,ri,rj,ri,rj,0,1,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaPairHopping[idx] * myTmp[i];
      /* modified by YN */
    }

    /* Exchange Coupling */
    #pragma omp for private(idx,ri,rj,myTmp) schedule(dynamic) nowait
    for(idx=0;idx<NExchangeCoupling;idx++) {
      ri = ExchangeCoupling[idx][0];
      rj = ExchangeCoupling[idx][1];
    
      /* modified by YN */
      GreenFunc2(myTmp,ri,rj,rj,ri,0,1,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaExchangeCoupling[idx] * myTmp[i];

      GreenFunc2(myTmp,ri,rj,rj,ri,1,0,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaExchangeCoupling[idx] * myTmp[i];
      /* modified by YN */
    }

    /* Inter All */
    #pragma omp for private(idx,ri,rj,s,rk,rl,t) schedule(dynamic) nowait
    for(idx=0;idx<NInterAll;idx++) {
      ri = InterAll[idx][0];
      rj = InterAll[idx][2];
      s  = InterAll[idx][3];
      rk = InterAll[idx][4];
      rl = InterAll[idx][6];
      t  = InterAll[idx][7];
      
      /* modified by YN */
      GreenFunc2(myTmp,ri,rj,rk,rl,s,t,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaInterAll[idx] * myTmp[i];
      /* modified by YN */
    }

    /* modified by YN */
    #pragma omp critical
    {
      for(i=0;i<nVMCSampleHidden2;i++) e[i] += myEnergy[i];
    }
    /* modified by YN */
  }

  ReleaseWorkSpaceThreadInt();
  ReleaseWorkSpaceThreadComplex();
  return; /* modified by YN */
}
