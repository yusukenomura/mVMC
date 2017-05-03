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
void CalculateHamiltonian_real(double complex *e, const double ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                               const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                               const double complex *thetaHidden1, const double complex *thetaHidden2);

void CalculateHamiltonian_real(double complex *e, const double ip, int *eleIdx, const int *eleCfg, int *eleNum, 
                               const int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                               const double complex *thetaHidden1, const double complex *thetaHidden2){
/* modified by YN */
  const int *n0 = eleNum;
  const int *n1 = eleNum + Nsite;
  double myTmp; /* added by YN */
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
  double  *myBuffer;
  double  *myEnergy, myEnergy_tmp; /* modified by YN */

  RequestWorkSpaceThreadInt(Nsize+Nsite2+NProj+2*NNeuronSample); /* modified by YN */
  RequestWorkSpaceThreadDouble(4*NVMCSampleHidden+NQPFull+2*Nsize); /* modified by YN */
  RequestWorkSpaceThreadComplex(2*NSizeTheta); /* added by YN */
  /* GreenFunc1: NQPFull, GreenFunc2: NQPFull+2*Nsize */
  for(i=0;i<2*NVMCSampleHidden;i++) e[i] = 0.0; /* added by YN */

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
    myThetaHiddenNew1 = GetWorkSpaceThreadComplex(NSizeTheta); 
    myThetaHiddenNew2 = GetWorkSpaceThreadComplex(NSizeTheta); 
    myEnergy = GetWorkSpaceThreadDouble(nVCMSampleHidden2);
    myTmp = GetWorkSpaceThreadDouble(nVCMSampleHidden2);
    /* added by YN */
    myBuffer = GetWorkSpaceThreadDouble(NQPFull+2*Nsize);

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
#ifdef _DEBUG
#pragma omp master
    printf("    Debug: CoulombIntra\n");
#endif
    /* CoulombIntra */
    #pragma omp for private(idx,ri) nowait
    for(idx=0;idx<NCoulombIntra;idx++) {
      ri = CoulombIntra[idx];
      myEnergy_tmp += ParaCoulombIntra[idx] * n0[ri] * n1[ri]; /* modified by YN */
    }

#ifdef _DEBUG
#pragma omp master
    printf("    Debug: CoulombInter\n");
#endif
    /* CoulombInter */
    #pragma omp for private(idx,ri,rj) nowait
    for(idx=0;idx<NCoulombInter;idx++) {
      ri = CoulombInter[idx][0];
      rj = CoulombInter[idx][1];
      myEnergy_tmp += ParaCoulombInter[idx] * (n0[ri]+n1[ri]) * (n0[rj]+n1[rj]); /* modified by YN */
    }

#ifdef _DEBUG
#pragma omp master
    printf("    Debug: HundCoupling\n");
#endif
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

#ifdef _DEBUG
#pragma omp master
    printf("    Debug: Transfer\n");
#endif
    /* Transfer */
#pragma omp for private(idx,ri,rj,s) schedule(dynamic) nowait
    for(idx=0;idx<NTransfer;idx++) {
      ri = Transfer[idx][0];
      rj = Transfer[idx][2];
      s  = Transfer[idx][3];
      
      /* modified by YN */
      GreenFunc1_real(myTmp,ri,rj,s,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew, 
                      hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                      thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] -= creal(ParaTransfer[idx]) * myTmp[i];
      /* modified by YN */
      /* Caution: negative sign */
    }

    #pragma omp master
    {StopTimer(71);StartTimer(72);}

#ifdef _DEBUG
#pragma omp master
    printf("    Debug: PairHopping\n");
#endif
    /* Pair Hopping */
    #pragma omp for private(idx,ri,rj) schedule(dynamic) nowait
    for(idx=0;idx<NPairHopping;idx++) {
      ri = PairHopping[idx][0];
      rj = PairHopping[idx][1];
    
      /* modified by YN */
      GreenFunc2_real(myTmp,ri,rj,ri,rj,0,1,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew, 
                      hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                      thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaPairHopping[idx] * myTmp[i];
      /* modified by YN */
    }

#ifdef _DEBUG
#pragma omp master
    printf("    Debug: ExchangeCoupling, NExchangeCoupling=%d\n", NExchangeCoulpling);
#endif
    /* Exchange Coupling */
    #pragma omp for private(idx,ri,rj,myTmp) schedule(dynamic) nowait
    for(idx=0;idx<NExchangeCoupling;idx++) {
      ri = ExchangeCoupling[idx][0];
      rj = ExchangeCoupling[idx][1];
    
      /* modified by YN */
      GreenFunc2_real(myTmp,ri,rj,rj,ri,0,1,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                      hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                      thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaExchangeCoupling[idx] * myTmp[i];
      GreenFunc2_real(myTmp,ri,rj,rj,ri,1,0,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew,
                      hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                      thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaExchangeCoupling[idx] * myTmp[i];
      /* modified by YN */
    }
    
#ifdef _DEBUG
#pragma omp master
    printf("    Debug: InterAll, NInterAll=%d\n", NInterAll);
#endif
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
      GreenFunc2_real(myTmp,ri,rj,rk,rl,s,t,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew, 
                      hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                      thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      for(i=0;i<nVMCSampleHidden2;i++) myEnergy[i] += ParaInterAll[idx] * myTmp[i];
      /* modified by YN */
    }

    #pragma omp master
    {StopTimer(72);}
#ifdef _DEBUG    
    printf("    Debug: myEnergy=%lf\n", myEnergy);
#endif
    /* modified by YN */
    #pragma omp critical
    {
      for(i=0;i<nVMCSampleHidden2;i++) e[i] += myEnergy[i] + myEnergy_tmp;
    }
    /* modified by YN */
  }
#ifdef _DEBUG
#pragma omp master
  printf("    Debug: Release\n");
#endif
  ReleaseWorkSpaceThreadInt();
  ReleaseWorkSpaceThreadDouble();
  ReleaseWorkSpaceThreadComplex(); /* added by YN */
#ifdef _DEBUG
#pragma omp master
  printf("    Debug: HamRealFinish\n", NInterAll);
#endif
  return; /* modified by YN */
}
