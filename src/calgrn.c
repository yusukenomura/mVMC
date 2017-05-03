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
 * Cauculate Green Functions
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

/* modified by YN */
void CalculateGreenFunc(const double w, const double complex ip, int *eleIdx, int *eleCfg, int *eleNum, 
                        int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                        const double complex *thetaHidden1, const double complex *thetaHidden2); 

void CalculateGreenFunc(const double w, const double complex ip, int *eleIdx, int *eleCfg, int *eleNum, 
                        int *eleProjCnt, const int *hiddenCfg1, const int *hiddenCfg2, 
                        const double complex *thetaHidden1, const double complex *thetaHidden2){ 
/* modified by YN */

  int idx,idx0,idx1;
  int ri,rj,s,rk,rl,t;
  double complex *myTmp; /* modified by YN */
  int *myEleIdx, *myEleNum, *myProjCntNew;
  /* added by YN */
  const int nVMCSampleHidden2 = nVMCSampleHidden*2
  int i; 
  int *myHiddenCfgNew1;
  int *myHiddenCfgNew2;
  double complex *myThetaHiddenNew1; 
  double complex *myThetaHiddenNew2; 
  /* added by YN */
  double complex *myBuffer;

  RequestWorkSpaceThreadInt(Nsize+Nsite2+NProj+2*NNeuronSample); /* modified by YN */
  RequestWorkSpaceThreadComplex(2*NSizeTheta+2*NVMCSampleHidden+NQPFull+2*Nsize); /* modified by KI */
  /* GreenFunc1: NQPFull, GreenFunc2: NQPFull+2*Nsize */

/* modified by YN */
#pragma omp parallel default(shared)\
  private(myEleIdx,myEleNum,myProjCntNew,myBuffer,myTmp,idx,i,\
          myHiddenCfgNew1,myHiddenCfgNew2,myThetaHiddenNew1,myThetaHiddenNew2) 
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
    myTmp = GetWorkSpaceThreadComplex(nVMCSampleHidden2); 
    /* added by YN */
    myBuffer = GetWorkSpaceThreadComplex(NQPFull+2*Nsize);

    #pragma loop noalias
    for(idx=0;idx<Nsize;idx++) myEleIdx[idx] = eleIdx[idx];
    #pragma loop noalias
    for(idx=0;idx<Nsite2;idx++) myEleNum[idx] = eleNum[idx];

    #pragma omp master
    {StartTimer(50);}

    #pragma omp for private(idx,ri,rj,s,myTmp) schedule(dynamic) nowait
    for(idx=0;idx<NCisAjs;idx++) {
      ri = CisAjsIdx[idx][0];
      rj = CisAjsIdx[idx][2];
      s  = CisAjsIdx[idx][3];
      /* modified by YN */
      GreenFunc1(myTmp,ri,rj,s,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew, 
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, 
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); 
      LocalCisAjs[idx] = 0.0; /* change */
      for(i=0;i<nVMCSampleHidden2,i++) LocalCisAjs[idx] += myTmp[i]; /* change */
      /* modified by YN */
    }

    #pragma omp master
    {StopTimer(50);StartTimer(51);}
    
    #pragma omp for private(idx,ri,rj,s,rk,rl,t,myTmp) schedule(dynamic)
    for(idx=0;idx<NCisAjsCktAltDC;idx++) {
      /*
      ri = CisAjsCktAltDCIdx[idx][0];
      rj = CisAjsCktAltDCIdx[idx][1];
      s  = CisAjsCktAltDCIdx[idx][2];
      rk = CisAjsCktAltDCIdx[idx][3];
      rl = CisAjsCktAltDCIdx[idx][4];
      t  = CisAjsCktAltDCIdx[idx][5];
      */
      ri = CisAjsCktAltDCIdx[idx][0];
      rj = CisAjsCktAltDCIdx[idx][2];
      s  = CisAjsCktAltDCIdx[idx][1];
      rk = CisAjsCktAltDCIdx[idx][4];
      rl = CisAjsCktAltDCIdx[idx][6];
      t  = CisAjsCktAltDCIdx[idx][5];

      /* modified by YN */
      GreenFunc2(myTmp,ri,rj,rk,rl,s,t,ip,myEleIdx,eleCfg,myEleNum,eleProjCnt,myProjCntNew, 
                 hiddenCfg1,myHiddenCfgNew1,hiddenCfg2,myHiddenCfgNew2, /* added by YN */
                 thetaHidden1,myThetaHiddenNew1,thetaHidden2,myThetaHiddenNew2,myBuffer); /* modified by YN */
      for(i=0;i<nVMCSampleHidden2,i++) PhysCisAjsCktAltDC[idx] += w*myTmp[i];
      /* modified by YN */
    }
    
    #pragma omp master
    {StopTimer(51);StartTimer(52);}

    #pragma omp for private(idx) nowait
    for(idx=0;idx<NCisAjs;idx++) {
      PhysCisAjs[idx] += w*LocalCisAjs[idx];
    }
    
    #pragma omp master
    {StopTimer(52);StartTimer(53);}

    #pragma omp for private(idx,idx0,idx1) nowait
    for(idx=0;idx<NCisAjsCktAlt;idx++) {
      idx0 = CisAjsCktAltIdx[idx][0];
      idx1 = CisAjsCktAltIdx[idx][1];
      PhysCisAjsCktAlt[idx] += w*LocalCisAjs[idx0]*conj(LocalCisAjs[idx1]);// TBC conj ok?
    }

    #pragma omp master
    {StopTimer(53);}
  }

  ReleaseWorkSpaceThreadInt();
  ReleaseWorkSpaceThreadComplex();
  return;
}
