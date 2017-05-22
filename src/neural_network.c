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
 * Artificial Neural Network 
 *-------------------------------------------------------------
 * by Yusuke Nomura
 *-------------------------------------------------------------*/


inline double LogHiddenWeightVal(const double complex *thetaHidden);
inline double LogHiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld);
inline double HiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld);
void CalcThetaHidden(double complex *thetaHidden, const int *eleNum, const int *hiddenCfg);
void UpdateThetaHidden(const int ri, const int rj, const int s, double complex *thetaHiddenNew, 
                       const double complex *thetaHiddenOld, const int *hiddenCfg);
void UpdateHiddenCfg(int *hiddenCfg, double complex *thetaHiddenNew, double complex *thetaHidden);
void CompleteHiddenPhysIntIdx();  



inline double LogHiddenWeightVal(const double complex *thetaHidden) {
  int idx;
  double z=0;
  for(idx=0;idx<NSizeTheta;idx++) {
    z += log(cosh(creal(thetaHidden[idx])));
  }
  return z;
}


inline double LogHiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld) {
  int idx;
  double z=0;
  for(idx=0;idx<NSizeTheta;idx++) {
    z += log(cosh(creal(thetaHiddenNew[idx]))) - log(cosh(creal(thetaHiddenOld[idx])));
  }
  return z;
}


inline double HiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld) {
  int idx;
  double z=0;
  for(idx=0;idx<NSizeTheta;idx++) {
    z += log(cosh(creal(thetaHiddenNew[idx]))) - log(cosh(creal(thetaHiddenOld[idx])));
  }
  return exp(z);
}



void CalcThetaHidden(double complex *thetaHidden, const int *eleNum, const int *hiddenCfg) {
  int f,i,j,k;
  int idx,rsi,offset1,offset2,offset3,offset4;
  double complex *tmpTheta;

  const int nSetHidden=NSetHidden;
  const int nSite=Nsite;
  const int nSetDeepHidden=NSetDeepHidden;
  const int nIntPerNeuron=NIntPerNeuron;
  const int nNeuronPerSet=NNeuronPerSet;

  for(f=0;f<nSetHidden;f++) { 
    tmpTheta = thetaHidden + f*nNeuronPerSet; 
    offset1 = f*nNeuronPerSet;
    offset2 = f*nIntPerNeuron;
    offset3 = offset2*nSetDeepHidden; // = f*nNeuronSample
    for(i=0;i<nNeuronPerSet;i++) { 
      idx = offset1 + i; 

      /* Magnetic field acting on Hidden variables */    
      tmpTheta[i] = HiddenMagField[f]; // TBC 

      /* Interaction between hidden and phyiscal variables  
         i-th neuron in f-th set has NIntPerNeuron interactions; through j-th interaction,
         it interacts with HiddenPhysIntIdx1[f*NNeuronPerSet+i][j]-th physical variable.    */
      for(j=0;j<nIntPerNeuron;j++) {
        rsi = HiddenPhysIntIdx1[idx][j]; 
        if( rsi > nSite-1 ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        if( abs(eleNum[rsi]-eleNum[rsi+nSite]) != 1 ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        tmpTheta[i] += HiddenPhysInt[offset2+j] * (double complex)(eleNum[rsi]-eleNum[rsi+nSite]); // TBC 
        for(k=0;k<nSetDeepHidden;k++) {
          offset4 = k*nIntPerNeuron;
          tmpTheta[i] += HiddenHiddenInt[offset3+offset4+j] * (double complex)(hiddenCfg[offset4+rsi]); // TBC 
        }
      }
    }
  } 

  return;
}


/* An electron with spin s hops from ri to rj. */
void UpdateThetaHidden(const int ri, const int rj, const int s, double complex *thetaHiddenNew, 
                       const double complex *thetaHiddenOld, const int *hiddenCfg) { 
  int f,i,j,rsi,rsj;
  int idx,offset1,offset2;
  double complex *tmpTheta;

  const int nSizeTheta=NSizeTheta;
  const int nSetHidden=NSetHidden;
  const int nIntPerNeuron=NIntPerNeuron;
  const int nNeuronPerSet=NNeuronPerSet;
  const int nSite=Nsite;
  
  if(thetaHiddenNew!=thetaHiddenOld) {
    for(idx=0;idx<nSizeTheta;idx++) thetaHiddenNew[idx] = thetaHiddenOld[idx];
  }
  if(ri==rj) return;

  /* comment: for the moment, hiddenCfg is not used */
  rsi = ri; //+ s*nSite;
  rsj = rj; //+ s*nSite; 
  for(f=0;f<nSetHidden;f++) { 
    tmpTheta = thetaHiddenNew + f*nNeuronPerSet; 
    offset1 = f*nNeuronPerSet;
    offset2 = f*nIntPerNeuron;
    for(i=0;i<nNeuronPerSet;i++) { 
      idx = offset1 + i; 

      /* Interaction between hidden and phyiscal variables  
         i-th neuron in f-th set interacts with rsi-th physical variable 
         through HiddenPhysIntIdx3[f*NNeuronPerSet+i][rsi]-th type of interaction. */
      j = HiddenPhysIntIdx3[idx][rsi]; 
      if( abs(1-2*s) != 1 ) MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
      tmpTheta[i] -= HiddenPhysInt[offset2+j]*(double complex)(1-2*s); // TBC 
      j = HiddenPhysIntIdx3[idx][rsj]; 
      tmpTheta[i] += HiddenPhysInt[offset2+j]*(double complex)(1-2*s); // TBC
    }
  }

  return; 
}


void UpdateHiddenCfg(int *hiddenCfg, double complex *thetaHiddenNew, double complex *thetaHidden){
  int i,j,k,dhi,f,idx;
  int offset1,offset3,offset4;
  double x; 
  int *tmpHiddenCfg;
  double complex *tmpTheta;

  const int nSizeTheta=NSizeTheta;
  const int nSetHidden=NSetHidden;
  const int nSetDeepHidden=NSetDeepHidden;
  const int nNeuronSample=NNeuronSample;
  const int nIntPerNeuron=NIntPerNeuron;
  const int nNeuronPerSet=NNeuronPerSet;

  if(thetaHiddenNew!=thetaHidden) {
    for(idx=0;idx<nSizeTheta;idx++) thetaHiddenNew[idx] = thetaHidden[idx];
  }

  Counter[4]++;  
  k = gen_rand32()%nSetDeepHidden;
  dhi = gen_rand32()%nIntPerNeuron;
  offset4 = k*nIntPerNeuron;
  tmpHiddenCfg = hiddenCfg + offset4;

  /* calculate new theta */
  for(f=0;f<nSetHidden;f++) { 
    tmpTheta = thetaHiddenNew + f*nNeuronPerSet; 
    offset1 = f*nNeuronPerSet;
    offset3 = f*nNeuronSample; // = f*nIntPerNeuron*nSetDeepHidden;

    for(i=0;i<nNeuronPerSet;i++) { 
      idx = offset1 + i; 

      /* Interaction between hidden and deep hidden variables  
         i-th neuron in f-th set interacts with dhi-th deep hidden neuron in k-th set 
         through (HiddenPhysIntIdx3[f*NNeuronPerSet+i][dhi]+f*NNeuronSample+k*nIntPerNeuron)-th type of interaction. */
      j = HiddenPhysIntIdx3[idx][dhi]; 
      tmpTheta[i] -= 2.0*HiddenHiddenInt[offset3+offset4+j] * (double complex)(tmpHiddenCfg[dhi]); // TBC
    }
  }
  x = HiddenWeightRatio(thetaHiddenNew,thetaHidden);
  x *= exp(-2.0*DeepHiddenMagField[k]*(double)(tmpHiddenCfg[dhi]));
  if( genrand_real2() < x ) {
    tmpHiddenCfg[dhi] *= -1;
    for(idx=0;idx<nSizeTheta;idx++) thetaHidden[idx] = thetaHiddenNew[idx];
    Counter[5]++;  
  }

  return;
}


void CompleteHiddenPhysIntIdx() {
  int i,j,f,rsi; 
  int offset1,offset2; 
  int *int_chk; /* just for check */ 
  FILE *file1,*file2,*file3; /* to be deleted */

/* 
  j-th type of interaction in f-th set connects i-th neuron with 
  HiddenPhysIntIdx2[f*NIntPerNeuron+j][i]-th physical variable.
*/ 
  for(f=1;f<NSetHidden;f++) {
    offset2 = f*NIntPerNeuron;  
    for(j=0;j<NIntPerNeuron;j++) {
    for(i=0;i<Nsite;i++) {
      HiddenPhysIntIdx2[offset2+j][i] = HiddenPhysIntIdx2[j][i];
    } 
    } 
  }
                         
/* 
 i-th neuron in f-th set has NIntPerNeuron interactions; through j-th interaction,
 it interacts with HiddenPhysIntIdx1[f*NNeuronPerSet+i][j]-th physical variable.     
                            
 i-th neuron in f-th set interacts with rsi-th physical variable 
 through HiddenPhysIntIdx3[f*NNeuronPerSet+i][rsi]-th type of interaction.
*/ 
  for(f=0;f<NSetHidden;f++) {
    offset1 = f*NNeuronPerSet;  
    offset2 = f*NIntPerNeuron;  
    for(i=0;i<NNeuronPerSet;i++) {
      for(j=0;j<NIntPerNeuron;j++) {
        HiddenPhysIntIdx1[offset1+i][j] = HiddenPhysIntIdx2[offset2+j][i];
        HiddenPhysIntIdx3[offset1+i][HiddenPhysIntIdx1[offset1+i][j]] = j; 
      } 
    }
  } 

/* start checking */
/* TBC */

  /* Here, we assume that Nsite = NIntPerNeuron, i.e., every neuron interacts with all the sites.
     If we consider more general interaction, this part should be modified */

  int_chk = (int*)malloc(sizeof(int)*Nsite);
  if( Nsite != NIntPerNeuron ) {
    fprintf(stderr, "  Nsite != NIntPerNeuron, not implemented .\n");
    MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
  }

  for(f=0;f<NSetHidden;f++) {
    offset1 = f*NNeuronPerSet;  
    offset2 = f*NIntPerNeuron;  

    /* check for HiddenPhysIntIdx1 */
    for(i=0;i<NNeuronPerSet;i++) { 
      for(j=0;j<NIntPerNeuron;j++) int_chk[j] = 0; 
      for(j=0;j<NIntPerNeuron;j++) int_chk[HiddenPhysIntIdx1[offset1+i][j]] += 1;
      for(j=0;j<NIntPerNeuron;j++) {
        if( int_chk[j] != 1 ){ 
          fprintf(stderr, "  HiddenPhysIntIdx1 is someting wrong .\n");
	  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        }
      }
    }

    /* check for HiddenPhysIntIdx2 */
    for(j=0;j<NIntPerNeuron;j++) {
      for(i=0;i<Nsite;i++) int_chk[i] = 0; 
      for(i=0;i<Nsite;i++) int_chk[HiddenPhysIntIdx2[offset2+j][i]] += 1;
      for(i=0;i<Nsite;i++) { 
        if( int_chk[i] != 1 ){ 
          fprintf(stderr, "  HiddenPhysIntIdx2 is someting wrong .\n");
	  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        }
      }
    }

    /* check for HiddenPhysIntIdx3 */
    for(i=0;i<NNeuronPerSet;i++) { 
      for(rsi=0;rsi<Nsite;rsi++) int_chk[rsi] = 0; 
      for(rsi=0;rsi<Nsite;rsi++) int_chk[HiddenPhysIntIdx3[offset1+i][rsi]] += 1;
      for(rsi=0;rsi<Nsite;rsi++) {
        if( int_chk[rsi] != 1 ){ 
          fprintf(stderr, "  HiddenPhysIntIdx3 is someting wrong .\n");
	  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        }
      }
    }

  } 
  free(int_chk); 
/* end checking */


/* start writing (to be deleted) */
  file1 = fopen("check_Idx1.txt","w");
  file2 = fopen("check_Idx2.txt","w");
  file3 = fopen("check_Idx3.txt","w");
  for(f=0;f<NSetHidden;f++) {
    offset1 = f*NNeuronPerSet;  
    offset2 = f*NIntPerNeuron;  
    for(i=0;i<NNeuronPerSet;i++) { 
      for(j=0;j<NIntPerNeuron;j++) fprintf(file1,"%d %d %d \n", i, j, HiddenPhysIntIdx1[offset1+i][j]);
    }
    for(j=0;j<NIntPerNeuron;j++) {
      for(i=0;i<Nsite;i++) fprintf(file2,"%d %d %d \n", j, i, HiddenPhysIntIdx2[offset2+j][i]);
    }
    for(i=0;i<NNeuronPerSet;i++) { 
      for(rsi=0;rsi<Nsite;rsi++) fprintf(file3,"%d %d %d \n", i, rsi, HiddenPhysIntIdx3[offset1+i][rsi]);
    }
  fprintf(file1,"\n");fprintf(file2,"\n");fprintf(file3,"\n");
  }
  fclose(file1); fclose(file2); fclose(file3); 
/* end writing (to be deleted) */

  return;
}
