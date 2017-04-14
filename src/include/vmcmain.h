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
 * main program header
 *-------------------------------------------------------------
 * by Satoshi Morita
 *-------------------------------------------------------------*/

#ifndef _VMC_INCLUDE_FILES
#define _VMC_INCLUDE_FILES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include <limits.h>
#include <string.h>
#include <complex.h>

#ifdef _mpi_use
  #include <mpi.h>
#else
typedef int MPI_Comm;
MPI_Comm MPI_COMM_WORLD=0;
inline void MPI_Init(int argc, char* argv[]) {return;}
inline void MPI_Finalize() {return;}
inline void MPI_Abort(MPI_Comm comm, int errorcode) {exit(errorcode); return;}
inline void MPI_Barrier(MPI_Comm comm) {return;}
inline void MPI_Comm_size(MPI_Comm comm, int *size) {*size = 1; return;}
inline void MPI_Comm_rank(MPI_Comm comm, int *rank) {*rank = 0; return;}
#endif /* _mpi_use */

extern int omp_get_max_threads(void);
extern int omp_get_thread_num(void);

#include "../sfmt/SFMT.h"
#include "version.h"
#include "global.h"

#include "../safempi.c"
#include "../safempi_fcmp.c"
#include "../time.c"
#include "../workspace.c"

#ifdef _lapack
 #include "../stcopt_dposv.c"
#else
 #include "../stcopt_pdposv.c"
#endif

#include "../gauleg.c"
#include "../legendrepoly.c"
#include "../splitloop.c"
#include "../avevar.c"
#include "../average.c"

#include "../parameter.c"
#include "../projection.c"
#include "../neural_network.c" /* added by YN */ 
#include "../slater.c"
#include "../qp.c"
#include "../qp_real.c"
#include "../matrix.c"
#include "../pfupdate.c"
#include "../pfupdate_real.c"
#include "../pfupdate_two_fcmp.c"
#include "../pfupdate_two_real.c"
#include "../locgrn.c"
#include "../locgrn_real.c"
#include "../calham.c"
#include "../calham_real.c"
#include "../calgrn.c"
//#include "lslocgrn.c" // ignoring Lanczos To be added
#include "../setmemory.c"
#include "../readdef.c"
#include "../initfile.c"

#include "../vmcmake.c"
#include "../vmcmake_real.c"
#include "../vmccal.c"

#endif /* _VMC_INCLUDE_FILES */
