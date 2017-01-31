// Distributed two-dimensional Discrete FFT transform
// Chris Turner
// ECE4122 Project 1
// Based on Additional Code from Dr. George F. Riley, Georgia Tech
//


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include <math.h>
#include <mpi.h>

#include "Complex.h"
#include "InputImage.h"

int rc, numtasks, rank, nCPUs, w, height;

using namespace std;
void Transform1D(Complex* h, int w, Complex* H, int rank, int nCPUs);


void transpose(int w, int h, Complex* origArr, Complex* resArr)
{ // Trnaspose function to perform matrix transpose.
  // Used to eliminated need for multiple fft transofrm functions.
  for (int j = 0; j < w; j++){
 	for (int i = 0; i < h; i++) {
	resArr[(i*w) + j]=origArr[(j*h) + i];
	}
  }
}



void Transform2D(const char* inputFN) 
{ 
  InputImage image(inputFN);  // Create the helper object for reading the image
  // Setup of variables, and prepare empty arrays for 1d transform.
  MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  height = image.GetHeight(); 
  w = image.GetWidth();
  Complex* h = image.GetImageData();
  image.SaveImageData("h.txt", h, w, height);
  Complex* Hform = new Complex [w * height];
  nCPUs = height/numtasks;
  // Each rank will only perform transofrm on a given block dependent on nCPUs
  // Perform 1d transform on rank based assignment.
  Transform1D(h, w, Hform, rank, nCPUs);
  // Send data back to rank 0 for combination and exporting.	
  Hform = Hform + (rank*w*nCPUs);
  Complex* Htran = new Complex [w * height];
  if(rank == 0) {
	for(int round = 1; round < numtasks; round++){
		MPI_Status status;
		rc = MPI_Recv((Hform + (height*w*round/nCPUs)), (height*w*2)/nCPUs, MPI_COMPLEX, round, 0, MPI_COMM_WORLD, &status);
		if (rc != MPI_SUCCESS) {
			cout << "Rank " << rank << " recv failed, rc " << rc << endl << "\n";
			MPI_Finalize();
		}
	}
  //image.SaveImageData("1d.txt", Hform, w, height); 	
  transpose( w, height, Hform, Htran);
  } else if ( rank != 0) {
		MPI_Request request;
		rc = MPI_Isend(Hform, (height*w*2)/nCPUs, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &request);
		if (rc != MPI_SUCCESS) {
			cout << "Rank " << rank << " recv failed, rc " << rc << endl << "\n";
			MPI_Finalize();
		}
  }
  if(rank==0) {
	for(int round = 1; round < numtasks; round++){
	MPI_Request request;
	MPI_Isend(Htran, (2*height*w), MPI_COMPLEX, round, 0, MPI_COMM_WORLD, &request);	
	}
  } else {
	MPI_Status status;
	MPI_Recv(Htran, (2*height*w), MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &status);
  }
  Transform1D(Htran, w, Hform, rank, nCPUs);
  Hform = Hform + (rank*w*nCPUs);
  if(rank == 0) {
	for(int round = 1; round < numtasks; round++){
		MPI_Status status;
		rc = MPI_Recv((Hform + (height*w*round/nCPUs)), (height*w*2)/nCPUs, MPI_COMPLEX, round, 0, MPI_COMM_WORLD, &status);
		if(rc != MPI_SUCCESS) {
			cout << "Rank " << rank << " recv failed, rc " << rc << endl << "\n";
			MPI_Finalize();
		}
	}
  transpose( w, height, Hform, Htran);
  image.SaveImageData("2d.txt", Htran, w, height);
  } else if ( rank != 0) {
	MPI_Request request;
	rc = MPI_Isend(Hform, (height*w*2)/nCPUs, MPI_COMPLEX, 0, 0, MPI_COMM_WORLD, &request);
	if (rc != MPI_SUCCESS) {
		cout << "Rank " << rank << " recv failed, rc " << rc << endl << "\n";
		MPI_Finalize();
	}
  }  
}



void Transform1D(Complex* h,int w,Complex* H,int proc,int rows )
{
  int tformbegin = proc*rows*w;
  for(int y = 0; y < rows; y++) { // y = row currently being transformed.
  	for (int N = 0; N < w; N++) {	// N = currently location being calculated.
		Complex sum = 0; 
		for (int k = 0; k < w; k++) { // k = little h value currently being calculated.
			Complex X(cos(2*M_PI*N*k/w),-sin(2*M_PI*N*k/w));				
			sum = sum + (X * h[tformbegin+(y*w)+k]); // Sum of all row components.
		}	
  		H[tformbegin+(y*w)+N] = sum;
	}
  }
}




int main(int argc, char** argv)
{
  string fn("Tower.txt"); // default file name 
  if (argc > 1) fn = string(argv[1]);  // if name specified on cmd line
  // MPI initialization here
  rc = MPI_Init(&argc, &argv);
  if (rc != MPI_SUCCESS) {
	printf ("Error starting MPI.\n");
	MPI_Abort(MPI_COMM_WORLD, rc);
  }
  Transform2D(fn.c_str()); // Perform the transform.
  // Finalize MPI here
  MPI_Finalize();

}  
  

  
