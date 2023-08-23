#include "boundary.h"
#include <stdio.h>

//grid is parallelised in the x direction

void boundarypsi(double *psi, int m, int n, int b, int h, int w)
{

  int i,j;

  //BCs on bottom edge

  for (i=b+1;i<=b+w-1;i++)
    {
      psi[i*(m+2)+0] = (double)(i-b);
    }

  for (i=b+w;i<=m;i++)
    {
      psi[i*(m+2)+0] = (double)(w);
    }

  //BCS on RHS

  for (j=1; j <= h; j++)
    {
      psi[(m+1)*(m+2)+j] = (double) w;
    }

  for (j=h+1;j<=h+w-1; j++)
    {
      psi[(m+1)*(m+2)+j]=(double)(w-j+h);
    }
}

void boundaryzet(double *zet, double *psi, int m, int n)
{
  int i,j;

  //set top/bottom BCs:

  for (i=1;i<m+1;i++)
    {
      zet[i*(m+2)+0]   = 2.0*(psi[i*(m+2)+1]-psi[i*(m+2)+0]);
      zet[i*(m+2)+n+1] = 2.0*(psi[i*(m+2)+n]-psi[i*(m+2)+n+1]);
    }

  //set left BCs:

  for (j=1;j<n+1;j++)
    {
      zet[0*(m+2)+j] = 2.0*(psi[1*(m+2)+j]-psi[0*(m+2)+j]);
    }

  //set right BCs

  for (j=1;j<n+1;j++)
    {
      zet[(m+1)*(m+2)+j] = 2.0*(psi[m*(m+2)+j]-psi[(m+1)*(m+2)+j]);
    }
}
