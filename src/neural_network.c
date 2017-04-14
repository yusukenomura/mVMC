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


void CalcTheta(double *tmpThetaHidden, const int *eleNum);
void CompleteHiddenPhysIntIdx();  

void CalcTheta(double *tmpThetaHidden, const int *eleNum) {
  const int *n0=eleNum;
  const int *n1=eleNum+Nsite;
  int idx,offset1,offset2;
  int ri,rj;
  /* optimization for Kei */
  const int nProj=NProj;
  const int nSite=Nsite;

  printf("CalcTheta \n");

  return;
}

void CompleteHiddenPhysIntIdx() {
  int i,j,f; 
  int offset1,offset2; 

/* 
  j-th type of interaction in f-th set connects i-th neuron with 
  HiddenPhysIntIdx2[f*NIntPerNeuron+j][i]-th physical variable.
*/ 
  for(f=1;f<NSetHidden;f++) {
    offset2=f*NIntPerNeuron;  
    for(j=0;j<NIntPerNeuron;j++) {
    for(i=0;i<Nsite2;i++) {
      HiddenPhysIntIdx2[offset2+j][i] = HiddenPhysIntIdx2[j][i];
    } 
    } 
  }
                         
/* 
 i-th neuron in f-th set has NIntPerNeuron interactions; through j-th interaction,
 it interacts with HiddenPhysIntIdx1[f*(Nsite*2)+i][j]-th physical variable.     
*/ 
  for(f=0;f<NSetHidden;f++) {
    offset1=f*Nsite2;  
    offset2=f*NIntPerNeuron;  
    for(i=0;i<Nsite2;i++) {
    for(j=0;j<NIntPerNeuron;j++) {
      HiddenPhysIntIdx1[offset1+i][j] = HiddenPhysIntIdx2[offset2+j][i];
    } 
    }
  } 

  return;
}
