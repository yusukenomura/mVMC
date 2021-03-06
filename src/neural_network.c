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

inline double complex cCosh(const double complex x);
inline double complex cTanh(const double complex x);
inline double complex LogHiddenWeightVal(const double complex *thetaHidden);
inline double complex LogHiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld);
inline double complex HiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld);
void CalcThetaHidden(double complex *thetaHidden, const int *eleNum);
void UpdateThetaHidden(const int ri, const int rj, const int s,
                       double complex *thetaHiddenNew, const double complex *thetaHiddenOld );
void CompleteHiddenPhysIntIdx();  

inline double complex cCosh(const double complex x) {
  double complex z=0;
  
  z = cexp(x)+cexp(-x);
  return 0.5*z;
}

inline double complex cTanh(const double complex x) {
  double complex z=0;
  
  z = (cexp(x)-cexp(-x))/(cexp(x)+cexp(-x));
  return z;
}

inline double complex LogHiddenWeightVal(const double complex *thetaHidden) {
  int idx;
  double complex z=0;
  for(idx=0;idx<NSizeTheta;idx++) {
    z += clog(cCosh(thetaHidden[idx]));
  }
  return z;
}

inline double complex LogHiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld) {
  int idx;
  double complex z=0;
  for(idx=0;idx<NSizeTheta;idx++) {
    z += clog(cCosh(thetaHiddenNew[idx])) - clog(cCosh(thetaHiddenOld[idx]));
  }
  return z;
}

inline double complex HiddenWeightRatio(const double complex *thetaHiddenNew, const double complex *thetaHiddenOld) {
  int idx;
  double complex z=0;
  for(idx=0;idx<NSizeTheta;idx++) {
    z += clog(cCosh(thetaHiddenNew[idx])) - clog(cCosh(thetaHiddenOld[idx]));
  }
  return cexp(z);
}

void CalcThetaHidden(double complex *thetaHidden, const int *eleNum) {
  int f,i,j;
  int idx,rsi,offset1,offset2;
  double complex *tmpTheta;

  const int nSetHidden=NSetHidden;
  const int nIntPerNeuron=NIntPerNeuron;
  const int nNeuronPerSet=NNeuronPerSet;

  for(f=0;f<nSetHidden;f++) { 
    tmpTheta = thetaHidden + f*nNeuronPerSet; 
    offset1 = f*nNeuronPerSet;
    offset2 = f*nIntPerNeuron;
    for(i=0;i<nNeuronPerSet;i++) { 
      idx = offset1 + i; 

      /* Magnetic field acting on Hidden variables */    
      tmpTheta[i] = HiddenMagField[f]; // TBC 

      /* Interaction between hidden and phyiscal variables  
         i-th neuron in f-th set has NIntPerNeuron interactions; through j-th interaction,
         it interacts with HiddenPhysIntIdx1[f*NNeuronPerSet+i][j]-th physical variable.    */
      for(j=0;j<nIntPerNeuron;j++) {
        rsi = HiddenPhysIntIdx1[idx][j]; 
        tmpTheta[i] += HiddenPhysInt[offset2+j] * (double complex)(2*eleNum[rsi]-1); // TBC 
      }
    }
  } 

  return;
}


/* An electron with spin s hops from ri to rj. */
void UpdateThetaHidden(const int ri, const int rj, const int s,
                       double complex *thetaHiddenNew, const double complex *thetaHiddenOld) { 
  int f,i,j,rsi,rsj;
  int idx,offset1,offset2;
  double complex *tmpTheta;

  const int nSizeTheta=NSizeTheta;
  const int nSetHidden=NSetHidden;
  const int nIntPerNeuron=NIntPerNeuron;
  const int nNeuronPerSet=NNeuronPerSet;
  const int nSite2=Nsite2;
  const int nSite=Nsite;
  
  if(thetaHiddenNew!=thetaHiddenOld) {
    for(idx=0;idx<nSizeTheta;idx++) thetaHiddenNew[idx] = thetaHiddenOld[idx];
  }
  if(ri==rj) return;

  rsi = ri + s*nSite;
  rsj = rj + s*nSite; 
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
      tmpTheta[i] -= 2.0*HiddenPhysInt[offset2+j]; // TBC 
      j = HiddenPhysIntIdx3[idx][rsj]; 
      tmpTheta[i] += 2.0*HiddenPhysInt[offset2+j]; // TBC
    }
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
    for(i=0;i<Nsite2;i++) {
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

  /* Here, we assume that Nsite2 = NIntPerNeuron, i.e., every neuron interacts with all the sites.
     If we consider more general interaction, this part should be modified */

  int_chk = (int*)malloc(sizeof(int)*Nsite2);
  if( Nsite2 != NIntPerNeuron ) {
    fprintf(stderr, "  Nsite2 != NIntPerNeuron, not implemented .\n");
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
      for(i=0;i<Nsite2;i++) int_chk[i] = 0; 
      for(i=0;i<Nsite2;i++) int_chk[HiddenPhysIntIdx2[offset2+j][i]] += 1;
      for(i=0;i<Nsite2;i++) { 
        if( int_chk[i] != 1 ){ 
          fprintf(stderr, "  HiddenPhysIntIdx2 is someting wrong .\n");
	  MPI_Abort(MPI_COMM_WORLD,EXIT_FAILURE);
        }
      }
    }

    /* check for HiddenPhysIntIdx3 */
    for(i=0;i<NNeuronPerSet;i++) { 
      for(rsi=0;rsi<Nsite2;rsi++) int_chk[rsi] = 0; 
      for(rsi=0;rsi<Nsite2;rsi++) int_chk[HiddenPhysIntIdx3[offset1+i][rsi]] += 1;
      for(rsi=0;rsi<Nsite2;rsi++) {
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
      for(i=0;i<Nsite2;i++) fprintf(file2,"%d %d %d \n", j, i, HiddenPhysIntIdx2[offset2+j][i]);
    }
    for(i=0;i<NNeuronPerSet;i++) { 
      for(rsi=0;rsi<Nsite2;rsi++) fprintf(file3,"%d %d %d \n", i, rsi, HiddenPhysIntIdx3[offset1+i][rsi]);
    }
  fprintf(file1,"\n");fprintf(file2,"\n");fprintf(file3,"\n");
  }
  fclose(file1); fclose(file2); fclose(file3); 
/* end writing (to be deleted) */

  return;
}
