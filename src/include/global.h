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
 * global variables
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

#ifndef _INCLUDE_GLOBAL
#define _INCLUDE_GLOBAL

#define D_FileNameMax 256

/***** definition *****/
char CDataFileHead[D_FileNameMax]; /* prefix of output files */
char CParaFileHead[D_FileNameMax]; /* prefix for optimized variational parameters */

int NVMCCalMode; /* calculation mode
                    0: optimization of variational paraneters,
                    1: calculation of expectation values */
int NLanczosMode; /* mode of the single Lanczos step
                     0: none, 1: only energy, 2: Green functions */

int NStoreO; /* choice of store O: 0-> normal other-> store  */

int NDataIdxStart; /* starting value of the file index */
int NDataQtySmp; /* the number of output files */

int Nsite; /* the number of sites */
int Ne;    /* the number of electrons with up spin */
int Nsize; /* the number of electrons = 2*Ne */
int Nsite2; /* 2*Nsite */

int NSPGaussLeg; /* the number of points for the Gauss-Legendre quadrature */
int NSPStot; /* S of Spin projection */
int NMPTrans; /* the number of quantum projection for translation and point group symmetry */
int NQPFull; /* the total number of quantum projection = NSPGaussLeg*NMPTrans*NQPTransOpt */
int NQPFix; /* for QPFixWeight NSPGaussLeg*NMPTranss */

int NSROptItrStep; /* the number of SR method steps */
int NSROptItrSmp; /* the number of SR method steps for calculation of average value */
int NSROptFixSmp; /* the number of SR method steps with fixed samples (1 is recommended) */
int NSROptWithShift; /* the number of SR method steps with constant shift for diagonal elements */ /* added by YN */

double DSROptRedCut; /* SR stabilizing factor for truncation of redundant directions */
double DSROptStaDel; /* SR stabiliaing factor for diagonal element modification */
double DSROptShiftRatio; /* SR stabiliaing factor for diagonal element modification (constant shift) */ /* added by YN */
double DSROptStepDt; /* step width of the SR method */

int NVMCWarmUp; /* Monte Carlo steps for warming up */
int NVMCInterval; /* sampling interval [MCS] */ 
int NVMCSample; /* the number of samples */
int NExUpdatePath; /* update by exchange hopping  0: off, 1: on */

int RndSeed; /* seed for pseudorandom number generator */
int NSplitSize; /* the number of inner MPI processes */
 
/* total length of def array */
int NTotalDefInt, NTotalDefDouble;

/* zlocspin.def */
int NLocSpn; /* the number of local spin */
int *LocSpn; /* [Nsite] */
/* local spin flag  0: local spin, 1: itinerant electron */

/* for Hamiltonian */
int NTransfer;
int **Transfer; /* [NTransfer][4] */
double complex*ParaTransfer;

int NCoulombIntra;
int *CoulombIntra; /* [NCoulombIntra] */
double *ParaCoulombIntra;

int NCoulombInter;
int **CoulombInter; /* [NCoulombInter][2] */
double *ParaCoulombInter;

int NHundCoupling;
int **HundCoupling; /* [NHundCoupling][2] */
double *ParaHundCoupling;

int NPairHopping;
int **PairHopping; /* [NPairHopping][2] */
double *ParaPairHopping;

int NExchangeCoupling;
int **ExchangeCoupling; /* [NExchangeCoupling][2] */
double *ParaExchangeCoupling;

int NInterAll;
int **InterAll; /* [NInterAll][8] */
double complex*ParaInterAll;

/* for variational parameters */
int NGutzwillerIdx, *GutzwillerIdx; /* [Nsite] */
int NJastrowIdx, **JastrowIdx; /* [Nsite][Nsite] */
int NDoublonHolon2siteIdx, **DoublonHolon2siteIdx; /* DoublonHolon2siteIdx[idx][2*Nsite] */
int NDoublonHolon4siteIdx, **DoublonHolon4siteIdx; /* DoublonHolon4siteIdx[idx][4*Nsite] */
int NOrbitalIdx, **OrbitalIdx; /* [Nsite][Nsite] */
int **OrbitalSgn; /* OrbitalSgn[Nsite][Nsite] = +1 or -1 */

/* added by YN */
/* variables for neural network */
int NSetHidden;          /* Number of the set of Hidden variables = Hidden neuron density       */
                         /* A set consists of magnetic field and hidden-phys interaction        */
int NHiddenMagField;     /* Total number of magnetic-field variabls in hidden layers            */
int NHiddenPhysInt;      /* Total number of (hidden layer)-(physical layer) interactions        */
int NNeuronPerSet;       /* Number of neurons per one set                                       */
int FlagNeuronTrans;     /* FlagNeuronTrans=1: use translational symmetry, 0: symmetry not used */
int NIntPerNeuron;       /* Number of hidden-phys interactions per one neuron                   */
int NHiddenVariable;     /* Total number of hidden variables = NHiddenMagField + NHiddenPhysInt */  
int NSizeTheta;          /* Number of theta angles per set =  NSetHidden * NNeuronPerSet        */  
int **HiddenPhysIntIdx1; /* HiddenPysIntIdx1[NSetHidden*NNeuronPerSet(=NSizeTheta)][NIntPerNeuron]                            
                            i-th neuron in f-th set has NIntPerNeuron interactions; through j-th interaction, 
                            it interacts with HiddenPhysIntIdx1[f*NNeuronPerSet+i][j]-th physical variable.  */ 
int **HiddenPhysIntIdx2; /* HiddenPysIntIdx2[NSetHidden*NIntPerNeuron(=NHiddenPhysInt)][Nsite*2] 
                            j-th type of interaction in f-th set connects i-th neuron with       
                            HiddenPhysIntIdx2[f*NIntPerNeuron+j][i]-th physical variable.      */ 
int **HiddenPhysIntIdx3; /* HiddenPysIntIdx3[NSetHidden*NNeuronPerset(=NSizeTheta)][Nsite*2]                            
                            i-th neuron in f-th set interacts with rsi-th physical variable 
                            through HiddenPhysIntIdx3[f*NNeuronPerSet+i][rsi]-th type of interaction.  */ 
 /* Note that HiddenPhysIntIdx 1 and 2 have the same information but the order of data is different  */
/* added by YN */

/* zqptransidx.def */
int NQPTrans, **QPTrans; /* [NQPTrans][Nsite] */
int **QPTransSgn; /* QPTransSgn[NQPTrans][NSite] = +1 or -1 */
double complex *ParaQPTrans;

/* zqpopttrans.def */
int NQPOptTrans, **QPOptTrans; /* [NQPOptTrans][Nsite] */
int **QPOptTransSgn; /* QPOptTransSgn[NQPOptTrans][NSite] = +1 or -1 */
double *ParaQPOptTrans;

/* for Green functions */
int NCisAjs,         **CisAjsIdx;         /* [NCisAjs][3] */
int NCisAjsCktAlt,   **CisAjsCktAltIdx;   /* [NCisAjsCktAlt][8] */
int NCisAjsCktAltDC, **CisAjsCktAltDCIdx; /* [NCisAjsCktAltDC][6] */

/* Optimization flag */
int *OptFlag; /* [NPara]  1: optimized, 0 or 2: fixed */
int AllComplexFlag;/* 0 -> all real variables, !=0-> including complex variables*/

/* flag for anti-periodic boundry condition */
int APFlag; /* 0: periodic, 1: anti-periodic */

/* flag for shift of correlation factors */
/* 0: no shift, 1: shift. Set in ReadDefFileIdxPara(). */
int FlagShiftGJ=0;
int FlagShiftDH2=0;
int FlagShiftDH4=0;

/* flag for OptTrans mode */
int FlagOptTrans=0;
/* flag for Binary mode */
/* output zvo_var.dat (FileVar) as binary data */
int FlagBinary=0;

/* flag for file flush */
int NFileFlushInterval=1;

/***** Variational Parameters *****/
int NPara; /* the total number of variational prameters NPara=NProj+NHiddenVariable+NSlater+NOptTrans */ /* modified by YN */
int NProj;    /* the number of correlation factor */
int NSlater;  /* the number of pair orbital (f_ij) = NOrbitalIdx */
int NOptTrans; /* the number of weights for OptTrans. This is used only for variatonal parameters */
               /* NOptTrans = 0 (not OptTrans mode) or NQPOptTrans (OptTrans mode) */
double complex *Para;   /* variatonal parameters */
double complex *Proj;   /* correlation factor (Proj    =Para) */
/* added by YN */
double complex *HiddenMagField; /* magnetic field acting on hidden neurons (HiddenMagField=Para+NProj)  */ 
double complex *HiddenPhysInt;  /* Interaction between hidden and physical variables (HiddenPhysInt=Para+NProj+NHiddenMagField)*/ 
/* added by YN */
double complex *Slater; /* pair orbital (Slater  =Para+NProj+NHiddenVariable) */         /* modified by YN */
double complex *OptTrans; /* weights    (OptTrans=Para+NProj+NHiddenVariable+NSlater) */ /* modified by YN */

/***** Electron Configuration ******/
int *EleIdx; /* EleIdx[sample][mi+si*Ne] */
int *EleCfg; /* EleCfg[sample][ri+si*Nsite] */
int *EleNum; /* EleIdx[sample][ri+si*Nsite] */
int *EleProjCnt; /* EleProjCnt[sample][proj] */
double *ThetaHidden; /* ThetaHidden[sample][NSizeTheta] = theta in Eq.(C2) in Carleo-Troyer Science */  /* added by YN */
double *logSqPfFullSlater; /* logSqPfFullSlater[sample] */

int *TmpEleIdx;
int *TmpEleCfg;
int *TmpEleNum;
int *TmpEleProjCnt;
double *TmpThetaHidden; /* added by YN */

int *BurnEleIdx;
int *BurnEleCfg;
int *BurnEleNum;
int *BurnEleProjCnt;
int BurnFlag=0; /* 0: off, 1: on */

/***** Slater Elements ******/
double complex *SlaterElm; /* SlaterElm[QPidx][ri+si*Nsite][rj+sj*Nsite] */
double complex *InvM; /* InvM[QPidx][mi+si*Ne][mj+sj*Ne] */
double complex *PfM; /* PfM[QPidx] */
// TBC only for real
double *SlaterElm_real; /* SlaterElm[QPidx][ri+si*Nsite][rj+sj*Nsite] */
double *InvM_real; /* InvM[QPidx][mi+si*Ne][mj+sj*Ne] */
double *PfM_real; /* PfM[QPidx] */

/***** Quantum Projection *****/
double complex *QPFullWeight; /* QPFullWeight[NQPFull] */
double complex *QPFixWeight; /* QPFixWeight[NQPFix] */
double complex *SPGLCos, *SPGLSin; /* [NSPGaussLeg]  cos(beta/2) and sin(beta/2) */
double complex *SPGLCosSin, *SPGLCosCos, *SPGLSinSin; /* [NSPGaussLeg] */

/***** Stocastic Reconfiguration *****/
int    SROptSize; /* 1+NPara */
double complex *SROptOO; /* [SROptSize*SROptSize] <O^\dagger O> */
double complex *SROptHO; /* [SROptSize]            < HO > */
double complex *SROptO;  /* [SROptSize] calculation buffar */
double complex *SROptO_Store;  /* [SROptSize*NVMCSample] calculation buffer */
//for real
double *SROptOO_real; /* [SROptSize*SROptSize] <O^\dagger O> */ //TBC
double *SROptHO_real; /* [SROptSize]            < HO > */       //TBC
double *SROptO_real;  /* [SROptSize] calculation buffar */      //TBC
double *SROptO_Store_real;  /* [SROptSize*NVMCSample] calculation buffer */



double complex *SROptData; /* [2+NPara] storage for energy and variational parameters */

/***** Physical Quantity *****/
double complex Wc; /* Weight for correlation sampling = <psi|x> */
double complex Etot; /* <H> */
double complex Etot2; /* <H^2> */

double complex *PhysCisAjs; /* [NCisAjs] */
double complex *PhysCisAjsCktAlt; /* [NCisAjsCktAlt] */
double complex *PhysCisAjsCktAltDC; /* [NCisAjsCktAltDC] */
double complex *LocalCisAjs; /* [NCisAjs] */

const int NLSHam = 2; /* 0: I, 1: H */
double complex *QQQQ; /* QQQQ[NLSHam][NLSHam][NLSHam][NLSHam]*/  //TBC
double complex *LSLQ; /* [NLSHam][NLSHam]*/                      //TBC   
 
double complex *QCisAjsQ; /* QCisAjsQ[NLSHam][NLSHam][NCisAjs]*/ //TBC
double complex *QCisAjsCktAltQ; /* QCisAjsCktAltQ[NLSHam][NLSHam][NCisAjsCktAlt]*/ //TBC
double complex *LSLCisAjs; /* [NLSHam][NCisAjs]*/                //TBC

/***** Output File *****/
/* FILE *FileCfg; */
FILE *FileOut;
FILE *FileVar;
FILE *FileTime;
FILE *FileSRinfo; /* zvo_SRinfo.dat */
FILE *FileCisAjs;
FILE *FileCisAjsCktAlt;
FILE *FileCisAjsCktAltDC;
FILE *FileLS;
FILE *FileLSQQQQ;
FILE *FileLSQCisAjsQ;
FILE *FileLSQCisAjsCktAltQ;

/* FILE *FileTimerList; */
/* FILE *FileOpt;    /\* zqp_opt *\/ */

/***** HitachiTimer *****/
const int NTimer=100;
double Timer[100], TimerStart[100];

/* flag for  SROptimization*/
int SRFlag; /* 0: periodic, 1: Diagonalization */

/***** openMP *****/
int NThread;

/***** for DGETRI and DSKPFA in CalculateMAll *****/
int LapackLWork;

/***** counter for vmcMake *****/
int Counter[4] = {0,0,0,0};
/* 0: hopping, 1: hopping accept, 2: exchange try, 3: exchange accept */

#endif /*  _INCLUDE_GLOBAL */
