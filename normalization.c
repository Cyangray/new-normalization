#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>    
#include <math.h>    
#include <fcntl.h>						/* Header for open/read/write */ 
#include <errno.h> 
#include <sys/types.h>
#include <sys/ioctl.h>          
#define MAXDIM    8192					/* Max dimension of Ex or Eg spectra */
#define BN0       9.154					/* Default Bn of 96Mo */
#define PI        3.14159265358979		/* The pi number */

FILE	*fp;
char	line[1024],cdum[128];
int		i, dimx, dimy, dim = MAXDIM-1, NchBn = 0, L1, L2, dim_sigext;
float	rho[MAXDIM], spincut[MAXDIM], sigpaw[MAXDIM], sigext[MAXDIM];
float   sig[MAXDIM],dsig[MAXDIM];
float	trans[MAXDIM], transext[MAXDIM], strength[MAXDIM];
int		ell;
float	Int = 0., Fac = 0., Eg, dum;
float	x1  = 0., x2 = 0., x3 = 0., x4 = 0., x5 = 0., x6 = 0., delta_x1;
float	eps = 1.e-20, x;
float   rho_ex(float);
float   sig_ex(float);
float   T_eg(float);
float   Bn_keV,ex,eg,de=10., Eres = 400. ; //de is steps for the integral of 10 keV, Eres=200 is Si+Nai resolution at rho(Ex=0)
int		makeroot1();

static void fgets_ignore(char *s, int size, FILE *stream)
{
    // suppress braindead and presumtuous glibc 'security' warning
    if( !fgets(s, size, stream) )
        return;
}


/* Defining defaults values, taken from 96Mo */
float	Bn = BN0, BnOld, a0 = 60., a1 = 120., It = 1.5;
float	D = 105., G = 150., D0 = 105., G0 = 150., D1 = 105., G1 = 150.;
int main()
{

	
	/* ************************************************** */
	/* Reading calibration and dimensions from: rhosp.rsg */
	/* ************************************************** */

	fp = fopen("rhosp.rsg", "r");
	if(fp == NULL){
		printf("No rhosp.rsg file found in your directory\n");
		exit(0);
	}
	else {
		fgets_ignore(line,sizeof(line),fp);
		fgets_ignore(line,sizeof(line),fp);
		fgets_ignore(line,sizeof(line),fp);
		fgets_ignore(line,sizeof(line),fp);
		fgets_ignore(line,sizeof(line),fp);
		fgets_ignore(line,sizeof(line),fp);
		fgets_ignore(line,sizeof(line),fp);
		sscanf(line,"%13s %7s %f %s %f",cdum, cdum, &a0, cdum, &a1);
		fgets_ignore(line,sizeof(line),fp);	
		fgets_ignore(line,sizeof(line),fp);
		sscanf(line,"%s %d %s %d",cdum, &dimx, cdum, &dimy);
		fgets_ignore(line,sizeof(line),fp);
		fclose(fp);
	}

	/* ***************************************** */
	/* Reading default values from previous runs */
	/* ***************************************** */
	fp = fopen("input.nrm", "r");
	if(fp == NULL){
		printf("\nCould not open file input.nrm, default values are used \n");
	}
	else {
		fgets_ignore(line,sizeof(line),fp);
		sscanf(line, " %d %f %f\n", &ell, &Bn, &It);
		fgets_ignore(line,sizeof(line),fp);
		sscanf(line, " %f %f \n", &D0, &G0);
		fgets_ignore(line,sizeof(line),fp);
		sscanf(line, " %f %f \n", &D1, &G1);
		fclose(fp);
	}
	
	/* ***************************************** */
	/* Reading default values from previous runs */
	/* ***************************************** */
	fp = fopen("input.cnt", "r");
	if(fp == NULL){
		printf("\nCould not open file input.cnt, default Bn value is used \n");
	}
	else {
		fgets_ignore(line,sizeof(line),fp);
		sscanf(line, " %f %f %f %f %f \n", &dum, &dum, &BnOld, &dum, &dum);
		if(Bn == BN0)Bn = BnOld;
		fclose(fp);
	}
  
	/* *********************** */
	/* Asking for input values */
	/* *********************** */
	
	fgets_ignore(line,sizeof(line),stdin);
/*	sscanf(line,"%d", &ell);*/
	if(ell == 0){
		D   = D0;
		G	= G0;
	}
	else{
		D   = D1;
		G	= G1;
	}
	

	fgets_ignore(line,sizeof(line),stdin);
/*	sscanf(line,"%f", &Bn);*/
	


	fgets_ignore(line,sizeof(line),stdin);
/*	sscanf(line,"%f", &It);*/
		

	fgets_ignore(line,sizeof(line),stdin);
/*	sscanf(line,"%f", &D);*/
  

	fgets_ignore(line,sizeof(line),stdin);
/*	sscanf(line,"%f", &G);*/

	/* ***********************************/
	/* Number of channels up to Bn or Bp */
	/* ***********************************/
        Bn_keV = Bn*1000.;
	NchBn = 1 + (int)(((Bn_keV-a0)/a1)+0.5);
	
	/* ***************************************************************** */
	/* Reading data of experimental nuclear level density: rhotmopaw.cnt */
	/* ***************************************************************** */

	fp = fopen("rhotmopaw.cnt", "r");
	if(fp == NULL){
		printf("No rhotmopaw.cnt file found in your directory\n");
		exit(0);
	}
	i = 0 ;
	while( i < dim){
        if(fgets(line,sizeof(line),fp) != NULL){sscanf(line,"%f", &rho[i]);}
	   i++; 
	}
	fclose(fp);
	
	/* ******************************************************************* */
	/* Reading data of extrapolated transmision coefficient: sigextpaw.cnt */
	/* ******************************************************************* */
	fp = fopen("sigextpaw.cnt", "r");
	if(fp == NULL){
		printf("No sigextpaw.cnt file found in your directory\n");
		exit(0);
	}
	i = 0 ;
	while(i < dim){
		if(fgets(line,sizeof(line),fp) != NULL){
            sscanf(line,"%f", &sigext[i]);
            }
		i++; 
	}
	fclose(fp);
	
	/* *************************************************************************** */
	/* Reading data and errors of experimental transmision coefficient: sigpaw.cnt */
	/* *************************************************************************** */
	fp = fopen("sigpaw.cnt", "r");
	if(fp == NULL){
		printf("No sigpaw.cnt file found in your directory\n");
		exit(0);
	}
	i = 0 ;
	while(i < MAXDIM && fgets(line,sizeof(line),fp) != NULL){
		sscanf(line,"%f", &sigpaw[i]);
		i++;
	}
	fclose(fp);
	dim = i/2;
    
    /* *************************** */
    /* Making output list in table */
    /* *************************** */
    for(i = 0; i < dim; i++){
        sig[i] = sigpaw[i];
        dsig[i]= sigpaw[i+dim];
    }
	
	/* **************************************************** */
	/* Finding L1 and L2 limits for region with data points */
	/* **************************************************** */
	L1 = 0.;
	L2 = dim;
	for(i = 0; i < dim; i++){
		if (sigpaw[i] > 10.*eps && L1 == 0) L1=i;
		if (sigpaw[i] < 10.*eps && i > dim/2 && L2 == dim ) L2=i-1;
	}
	
	/* ****************************************** */
	/* Reading spincutoff parameters: spincut.cnt */
	/* ****************************************** */
	fp = fopen("spincut.cnt", "r");
	if(fp == NULL){
		printf("No spincut.cnt file found in your directory\n");
		exit(0);
	}
	i = 0 ;
	while(i < NchBn){
        if(fgets(line,sizeof(line),fp) != NULL){sscanf(line,"%f", &spincut[i]);}
		spincut[i]=2.*spincut[i]*spincut[i];
	   i++; 
	}
	fclose(fp);
	
    
	/* *************************************** */
	/* Calculating integral, fasten seat-belts */
	/* *************************************** */
	if( ell == 0){				/* s-wave (l=0) capture */ 
		if(It == 0.0){			/*I_i = 1/2 => I_f = 1/2, 3/2 */
            while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
                
// printf("Eg = %6.0f   T = %14.7f     Ex = %6.0f  Rho = %14.7f  Sig = %14.7f  x2 = %14.7f\n",eg, T_eg(eg),ex, rho_ex(ex),sig_ex(ex),T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex)) );

                
                eg = eg + de;
			}
			Int = x1 + x2;
		}
		if(It == 0.5){			/*I_i = 0, 1 => I_f = 0, 1, 2 */
			while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
                x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-0.) / sig_ex(ex)) * exp(-(It-0.)*(It-0.) / sig_ex(ex));
                x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
                x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = x1 + 2.*x2 + x3;
		}
		if(It == 1.0){			/*I_i = 1/2, 3/2 => I_f = 1/2, 3/2, 5/2 */
            while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-0.) / sig_ex(ex)) * exp(-(It-0.)*(It-0.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = 2.*x1 + 2.*x2 + x3;
		}
		if(It > 1.0){			/*I_i = It+1/2, It-1/2 => I_f = It-3/2, It-1/2, It+1/2, It+3/2 */
            while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
                x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-1.) / sig_ex(ex)) * exp(-(It-1.)*(It-1.) / sig_ex(ex));
                x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It-0.) / sig_ex(ex)) * exp(-(It-0.)*(It-0.) / sig_ex(ex));
                x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
                x4 = x4 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
                eg = eg + de;
//                delta_x1= T_eg(eg)*rho_ex(ex) * ((It-1.) / sig_ex(ex)) * exp(-(It-1.)*(It-1.) / sig_ex(ex));
//                if(eg==(int)(eg/100.)*100.)printf("eg = %6.1f  ex = %6.1f T_eg = %10.3e rho_ex= %10.3e sig_ex= %7.2f delta_x1= %10.3e\n",eg,ex,T_eg(eg),rho_ex(ex),sig_ex(ex),delta_x1 );
			}
			Int = x1 + 2.*x2 + 2.*x3 + x4;
		}
	}

	if( ell == 1){				/* p-wave (l=1) capture */ 
		if(It == 0.0){			/*I_i = 1/2, 3/2 => I_f = 1/2, 3/2, 5/2 */
			while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
				x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+3.) / sig_ex(ex)) * exp(-(It+3.)*(It+3.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = 2.*x1 + 2.*x2 + x3;
		}
		if(It == 0.5){			/*I_i = 0, 1, 2 => I_f = 0, 1, 2, 3 */
			while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-0.) / sig_ex(ex)) * exp(-(It-0.)*(It-0.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
				x4 = x4 + T_eg(eg)*rho_ex(ex) * ((It+3.) / sig_ex(ex)) * exp(-(It+3.)*(It+3.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = x1 + 3.*x2 + 2.*x3 + x4;
		}
		if(It == 1.0){			/*I_i = 1/2, 3/2, 5/2 => I_f = 1/2, 3/2, 5/2, 7/2 */
            while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-0.) / sig_ex(ex)) * exp(-(It-0.)*(It-0.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
				x4 = x4 + T_eg(eg)*rho_ex(ex) * ((It+3.) / sig_ex(ex)) * exp(-(It+3.)*(It+3.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = 2.*x1 + 3.*x2 + 2.*x3 + x4;
		}
		if(It == 1.5){			/*I_i = 0, 1, 2, 3 => I_f = 0, 1, 2, 3, 4 */
            while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-1.) / sig_ex(ex)) * exp(-(It-1.)*(It-1.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+0.) / sig_ex(ex)) * exp(-(It+0.)*(It+0.) / sig_ex(ex));
				x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x4 = x4 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
				x5 = x5 + T_eg(eg)*rho_ex(ex) * ((It+3.) / sig_ex(ex)) * exp(-(It+3.)*(It+3.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = 1.*x1 + 3.*x2 + 3.*x3 + 2.*x4 + x5;
		}
		if(It == 2.0){			/*I_i = 1/2, 3/2, 5/2, 7/2 => I_f = 1/2, 3/2, 5/2, 7/2, 9/2 */
            while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-1.) / sig_ex(ex)) * exp(-(It-1.)*(It-1.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It+0.) / sig_ex(ex)) * exp(-(It+0.)*(It+0.) / sig_ex(ex));
				x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x4 = x4 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
				x5 = x5 + T_eg(eg)*rho_ex(ex) * ((It+3.) / sig_ex(ex)) * exp(-(It+3.)*(It+3.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = 2.*x1 + 3.*x2 + 3.*x3 + 2.*x4 + x5;
		}
		if(It > 2.0){			/*I_i = It-3/2, It-1/2, It+1/2, It+3/2 => I_f = It-5/2, It-3/2, It-1/2, It+1/2, It+3/2, It+5/2 */
            while (eg < Bn_keV+Eres){
                ex = Bn_keV - eg;
				x1 = x1 + T_eg(eg)*rho_ex(ex) * ((It-2.) / sig_ex(ex)) * exp(-(It-2.)*(It-2.) / sig_ex(ex));
				x2 = x2 + T_eg(eg)*rho_ex(ex) * ((It-1.) / sig_ex(ex)) * exp(-(It-1.)*(It-1.) / sig_ex(ex));
				x3 = x3 + T_eg(eg)*rho_ex(ex) * ((It+0.) / sig_ex(ex)) * exp(-(It+0.)*(It+0.) / sig_ex(ex));
				x4 = x4 + T_eg(eg)*rho_ex(ex) * ((It+1.) / sig_ex(ex)) * exp(-(It+1.)*(It+1.) / sig_ex(ex));
				x5 = x5 + T_eg(eg)*rho_ex(ex) * ((It+2.) / sig_ex(ex)) * exp(-(It+2.)*(It+2.) / sig_ex(ex));
				x6 = x6 + T_eg(eg)*rho_ex(ex) * ((It+3.) / sig_ex(ex)) * exp(-(It+3.)*(It+3.) / sig_ex(ex));
                eg = eg + de;
			}
			Int = x1 + 2.*x2 + 3.*x3 + 3.*x4 + 2.*x5 + x6;
		}
	}
    Int = Int * de;
	Fac = Int * D / G;									/* Units = a1*D/G = keV*eV/(MeV*meV) = 1 */
	
			
	/* ************* */
	/* Final results */
	/* ************* */
 
 	/* **************************************************** */
	/* Storing default values for the next run in input.nrm */
	/* **************************************************** */
	fp = fopen("input.nrm", "w");
	if(fp == NULL){
		printf("Could not open file input.nrm \n");
		exit(0);
	}
	else {
	if(ell == 0){
		D0  = D;
		G0	= G;
	}
	else{
		D1  = D;
		G1  = G;
	}
		fprintf(fp, " %d %f %f\n", ell, Bn, It);
        fprintf(fp, " %f %f \n", D0, G0);
		fprintf(fp, " %f %f \n", D1, G1);
        fprintf(fp, " %e \n", Fac);
	}
	fclose(fp);

	/* ******************************************************** */
	/* Normalized gamma strength function f(Eg) written to disk */
	/* ******************************************************** */
	fp = fopen("strength.nrm", "w");
	if(fp == NULL){
		printf("Could not open file strength.nrm \n");
		exit(0);
	}
	
	else {
		for (i = 0; i < dim; i++){
			Eg = (a0 + a1*(float)i)/1000.;
            strength[i] = 0;
			if(Eg > 0.) strength[i] = sigpaw[i]/(Fac*pow(Eg,3.0));
			if(strength[i] < eps ) strength[i] = 0;
			fprintf(fp, " %14.7e \n", strength[i]);
		}
		for (i = dim; i < 2*dim; i++){
			Eg = (a0 + a1*(float)(i-dim))/1000.;
            strength[i] = 0;
			if(Eg > 0.)strength[i] = sigpaw[i]/(Fac*pow(Eg,3.0));
			if(strength[i] < eps ) strength[i] = 0;
            if(strength[i-dim] < eps ) strength[i] = 0;
			fprintf(fp, " %14.7e \n", strength[i]);
		}
	}
	fclose(fp);	
	
	/* *************************************************************** */
	/* Normalized gamma-transmission coefficient T(Eg) written to disk */
	/* *************************************************************** */
	fp = fopen("trans.nrm", "w");
	if(fp == NULL){
		printf("Could not open file trans.nrm \n");
		exit(0);
	}
	else {
		for (i = 0; i < dim; i++){
			trans[i] = 2.*PI*sigpaw[i]/Fac;
			if(trans[i] < eps ) trans[i] = 0;
			fprintf(fp, " %14.7e \n", trans[i]);
		}
		for (i = dim; i < 2*dim; i++){
			trans[i] = 2.*PI*sigpaw[i]/Fac;
			if(trans[i] < eps ) trans[i] = 0;
			fprintf(fp, " %14.7e \n", trans[i]);
		}
	}
	fclose(fp);	
	
	/* ************************************************************************ */
	/* Writing normalized extended gamma-transmission coefficient T(Eg) to disk */
	/* ************************************************************************ */
	fp = fopen("transext.nrm", "w");
	if(fp == NULL){
		printf("Could not open file transext.nrm \n");
		exit(0);
	}
	else {
		for (i = 0; i < dim*10; i++){
			transext[i] = 2.*PI*sigext[i]/Fac;
            if(transext[i] < eps ) transext[i] = 0;
			fprintf(fp, " %14.7e \n", transext[i]);
		}
	}
	fclose(fp);	
	
	makeroot1();
	return(0);
}

float rho_ex(float ex)
{
    float ex1, ex2, rhox;
    int   ii,i1, i2 = -1;
    
    for (ii = 0; ii < NchBn; ii++){
        if (ex > a0 + a1*(float)ii){
            i2 = ii + 1;                 //just past energy
        }
    }
    if (i2 == -1 || i2 > NchBn-1){
        i2 = NchBn-1;
    }
    if (i2 == 0){
        i2 = 1;
    }
    i1  = i2 - 1;
    ex1 = a0 + a1*(float)i1;
    ex2 = a0 + a1*(float)i2;
    
    rhox = rho[i1] + (rho[i2] - rho[i1])*((ex - ex1)/(ex2-ex1));
    if (rhox < 0.)rhox = 0.;
    
    return rhox;
}

float sig_ex(float ex)
{
    float ex1, ex2, sigx;
    int   ii, i1, i2 = -1;
    
    for (ii = 0; ii < NchBn; ii++){
        if (ex > a0 + a1*(float)ii){
            i2 = ii + 1;                 //just past energy
        }
    }
    if (i2 == -1 || i2 > NchBn-1){
        i2 = NchBn-1;
    }
    if (i2 == 0){
        i2 = 1;
    }
    i1  = i2 - 1;
    ex1 = a0 + a1*(float)i1;
    ex2 = a0 + a1*(float)i2;
    
    sigx = spincut[i1] + (spincut[i2] - spincut[i1])*((ex - ex1)/(ex2-ex1));
    if (sigx < 0.01)sigx = 0.01;
    return sigx;
}

float T_eg(float eg)
{
    float eg1, eg2, Teg;
    int   ii, i1, i2 = -1;
    
    for (ii = 0; ii < NchBn; ii++){
        if (eg > a0 + a1*(float)ii){
            i2 = ii + 1;                 //just past energy
        }
    }
    if (i2 == -1 || i2 > NchBn-1){
        i2 = NchBn-1;
    }
    if (i2 == 0){
        i2 = 1;
    }
    i1  = i2 - 1;
    eg1 = a0 + a1*(float)i1;
    eg2 = a0 + a1*(float)i2;
    
    Teg = sigext[i1] + (sigext[i2] - sigext[i1])*((eg - eg1)/(eg2-eg1));
    if (Teg < 0.00000000000001)Teg = 0.00000000000001;
    return Teg;
}


int makeroot1(){
	float Emax, half, exx, lower, Tmin, Tmax;
	Emax = (a0 + a1*(dim-1))/1000.;
	Tmin = strength[L1];
	Tmax = strength[L2];
	for (i = 0; i < dim; i++){
	   if(strength[i] > Tmax ) Tmax = strength[i];
	   if(strength[i] < Tmin && strength[i] > 10.*eps) Tmin = strength[i];
	}
	
	Tmin = Tmin/10.;
	Tmax = Tmax*10.;
	exx = log10(Tmax) + log10(Tmin);
	half = pow(10.,exx/2.);
	lower = Tmin*0.6;

	fp = fopen("strength.cpp", "w+");
	if(fp == NULL){
		printf("Could not open file strength.cpp \n");
		exit(0);
	}
	else {
		
		fprintf(fp,"{\n");
		fprintf(fp,"	gROOT->Reset();\n");
		fprintf(fp,"	gROOT->SetStyle(\"Plain\");\n");
		fprintf(fp,"	gStyle->SetOptTitle(0);\n");
		fprintf(fp,"	gStyle->SetOptStat(0);\n");
		fprintf(fp,"	gStyle->SetFillColor(0);\n");
		fprintf(fp,"	gStyle->SetPadBorderMode(0);\n");
		fprintf(fp,"	m = (TH1F*)gROOT->FindObject(\"h\");\n");
		fprintf(fp,"	if (m) m->Delete();\n");
		fprintf(fp,"	TCanvas *c1 = new TCanvas(\"c1\",\"Gamma-ray strength function\",600,600);\n");	
		fprintf(fp,"	TH2F *h = new TH2F(\"h\",\" \",10,0.0,%8.3f,10,%9.3e,%9.3e);\n",Emax+0.5,Tmin,Tmax);
		fprintf(fp,"	ifstream strengthfile(\"strength.nrm\");\n");
		fprintf(fp,"	float strength[%d],strengtherr[%d],energyerr[%d];\n",dim+1,dim+1,dim+1);
        fprintf(fp,"	float energy[%d],trans[%d];\n",10*dim+1,10*dim+1);
		fprintf(fp,"	int i = 0;\n");
		fprintf(fp,"   float a0 = %8.4f;\n",a0/1000.); 
		fprintf(fp,"   float a1 = %8.4f;\n",a1/1000.);
		fprintf(fp,"	float x;	\n");
		fprintf(fp,"	while(strengthfile){\n");
		fprintf(fp,"		strengthfile >> x;\n");
		fprintf(fp,"		if(i<%d){\n",dim);
		fprintf(fp,"			strength[i] = x;\n");
		fprintf(fp,"			energy[i] = a0 + (a1*i);\n");
		fprintf(fp,"			energyerr[i] = 0.0;\n");
		fprintf(fp,"		}	\n");
		fprintf(fp,"		else{strengtherr[i-%d] = x;}\n",dim);
		fprintf(fp,"		i++;\n");
		fprintf(fp,"	}\n");
		fprintf(fp,"	TGraphErrors *strengthexp = new TGraphErrors(%d,energy,strength,energyerr,strengtherr);\n",dim+1);
        fprintf(fp,"    i = 0;\n");
        fprintf(fp,"    ifstream transfile(\"transext.nrm\");\n");
        fprintf(fp,"    while(transfile){\n");
        fprintf(fp,"        transfile >> x;\n");
        fprintf(fp,"        energy[i] = a0 + (a1*i);\n");
        fprintf(fp,"        trans[i] = x/(2.*3.14*energy[i]*energy[i]*energy[i]);\n");
        fprintf(fp,"        i++;\n");
        fprintf(fp,"    }\n");
        fprintf(fp,"    TGraph *rsfext = new TGraph(i,energy,trans);\n");
		fprintf(fp,"	c1->SetLogy();\n");
		fprintf(fp,"	c1->SetLeftMargin(0.14);\n");
		fprintf(fp,"	h->GetXaxis()->CenterTitle();\n");
		fprintf(fp,"	h->GetXaxis()->SetTitle(\"#gamma-ray energy E_{#gamma} (MeV)\");\n");
		fprintf(fp,"	h->GetYaxis()->CenterTitle();\n");
		fprintf(fp,"	h->GetYaxis()->SetTitleOffset(1.4);\n");
		fprintf(fp,"	h->GetYaxis()->SetTitle(\"#gamma-ray strength function (MeV^{-3})\");\n");
		fprintf(fp,"	h->Draw();\n");
		fprintf(fp,"	strengthexp->SetMarkerStyle(21);\n");
		fprintf(fp,"	strengthexp->SetMarkerSize(0.8);\n");
		fprintf(fp,"	strengthexp->Draw(\"P\");\n");
        fprintf(fp,"    rsfext->SetLineColor(4);\n");
        fprintf(fp,"    rsfext->SetLineWidth(2);\n");
        fprintf(fp,"    rsfext->Draw(\"L\");\n");
		fprintf(fp,"	TLatex t;\n");
		fprintf(fp,"	t.SetTextSize(0.05);\n");
		fprintf(fp,"	t.DrawLatex(%9.3f,%9.3e,\"^{xx}Yy\");\n",Emax/5.,0.3*Tmax);
		fprintf(fp,"	c1->Update();\n");
		fprintf(fp,"	c1->Print(\"strength.pdf\");\n");
		fprintf(fp,"	c1->Print(\"strength.eps\");\n");
		fprintf(fp,"	c1->Print(\"strength.ps\");\n");
		fprintf(fp,"}\n");
	}

	fclose(fp);
	return 0;
}
