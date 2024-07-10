#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>    
#include <math.h>    
#include <fcntl.h>					/* Header for open/read/write */ 
#include <errno.h>
#include <time.h>					/* Header for system time */
#include <sys/types.h>          
#include <sys/ioctl.h>
#define MAXLEV        10000			/* Max number of known levels */
#define MAXDIM        8192 			/* Max dimension of Ex or Eg spectra */
#define MAXENERGY     100.0			/* Max energy Ex or Eg (MeV)  */
 
char  line[1024];
char  cdum[128];
float rho[MAXDIM], drho[MAXDIM], levenergy[MAXLEV],rholev[MAXDIM];
float sig[MAXDIM], dsig[MAXDIM], nsig[MAXDIM], ndsig[MAXDIM], nsigL[MAXDIM], nsigH[MAXDIM];
int   dimRhox, dimRhoy, dimSigx, dimSigy, diml, dim, H, sigdimBn, dimmax;
float a0, a1, emin, emax, ex, eg;
float Anorm=1., alpha=0.;                 /* Normalization constants */
float eps = 1.e-20, x;
float eps_0 = 0.000;
float c1, c2, e1, e2; 
int   Lm, Hm;
FILE  *fp;
int   i,j,l;
/* Defining defaults values, taken from 162Dy */
float Amass = 164., Delta = 1.847, Bn = 8.197, rho0 = 3630000., drho0 = 363000.;
int   L1 = 10, L2 = 15, H1 = 49, H2 = 56;
int   TL1 = 10, TL2 = 15, TH1 = 49, TH2 = 56, Tfirst = 10, Tlast = 56;
int   isig = 1, itemp = 2, imodel = 1, ansL=0, ansH=0;
int   i0, i1, i2, FGmax;
float a, T, E1, E0, C1 = -1.296, eta, rhox, sig2, spcu;
float aRobin = 16.824, E1Robin = 0.505;
float TRobin = 0.5, E0Robin = 0.0;
float abestL = -1000.,bbestL = -1000.,abestH = -1000.,bbestH = -1000.;
float red = 1.;
float b1, b2;
float exe1=0.5, sige1=2.7, exe2=-1, sige2=7.7;
double rhoobs[MAXDIM],sigma2[MAXDIM];
double start_spin,stop_spin=30.;
float part_res=0;

int   searchAalpha();
int   searchT();
int   makeroot1();
int   makeroot2();
int   makeroot3();
void  rhofg(float Amass, float ex, float a, float T, float E1, float E0, int isig, int itemp, int imodel, float red, float b1, float b2);
float corrL();
float corrH();
float anchorL();
float anchorH();
int   extendL();
int   extendH();
float Tlow, Thigh, Tx, dTx, E0x, chi2_lowx, chi2_highx;
void  nld_talys();
void  ReadSmooth(float); // Read and smooth discrete states...
double rhoexp(double);
double spin_distribution(double, double);
double temperature(double);

float a0_x, a1_x, a2_x;
float a0_y, a1_y, a2_y;
int   dim_x, dim_y;
float fg[512][512];


static void fgets_ignore(char *s, int size, FILE *stream)
{
    // suppress braindead and presumtuous glibc 'security' warning
    if( !fgets(s, size, stream) )
        return;
}

int main()
{

    /* ************************************************************* */
    /* Reading calibration and dimensions from: rhosp.rsg            */
    /* ************************************************************* */
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
        sscanf(line,"%s %d %s %d",cdum, &dimRhox, cdum, &dimRhoy);
        fgets_ignore(line,sizeof(line),fp);
        fclose(fp);
    }
    
    /* ************************************************************* */
    /* Reading calibration and dimensions from: sigsp.rsg            */
    /* ************************************************************* */
    fp = fopen("sigsp.rsg", "r");
    if(fp == NULL){
        printf("No sigsp.rsg file found in your directory\n");
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
        sscanf(line,"%s %d %s %d",cdum, &dimSigx, cdum, &dimSigy);
        fgets_ignore(line,sizeof(line),fp);
        fclose(fp);
    }
	
    /* ************************************************************************* */
    /* Reading data and errors of experimental nuclear level density: rhopaw.rsg */
    /* ************************************************************************* */
    fp = fopen("rhopaw.rsg", "r");
    if(fp == NULL){
        printf("No rhopaw.rsg file found in your directory\n");
        exit(0);
    }
    l = 0 ;
    while( l <= dimRhox){
        if(fgets(line,sizeof(line),fp) != NULL){sscanf(line,"%f", &rho[l]);}
        l++; 
    }
    l = 0 ;
    while(l <= dimRhox){
        if(fgets(line,sizeof(line),fp) != NULL){sscanf(line,"%f", &drho[l]);}
        l++; 
    }
    fclose(fp);
	
    /* *************************************************************************** */
    /* Reading data and errors of experimental transmision coefficient: sigpaw.rsg */
    /* *************************************************************************** */
    fp = fopen("sigpaw.rsg", "r");
    if(fp == NULL){
        printf("No sigpaw.rsg file found in your directory\n");
        exit(0);
    }
    l = 0 ;
    while( l <= dimSigx){
        if(fgets(line,sizeof(line),fp) != NULL){sscanf(line,"%f", &sig[l]);}
        l++; 
    }
    l = 0 ;
    while( l <= dimSigx){
        if(fgets(line,sizeof(line),fp) != NULL){sscanf(line,"%f", &dsig[l]);}
        l++; 
    }
    fclose(fp);
	
    /* *************************************** */
    /* Finding common upper limits for spectra */
    /* *************************************** */
    H = 0;
    dim = 0;
    dimmax=dimRhox;
    if(dimmax < dimSigx)dimmax=dimSigx;
    l = (int)fabs(a0/a1);
    for (i = l; i <= dimmax; i++){
        if(rho[i]==0 && drho[i]==0 && H == 0) H = i;	
        if(rho[i]==0 && drho[i]==0 && sig[i]==0 && dsig[i]==0 && dim == 0)dim = i;	
    }
    if(H == 0) H = dimmax;
    if(dim == 0) dim = dimmax;
	
    /* ********************************************************* */
    /* Reading excitation energies of known levels: counting.dat */
    /* ********************************************************* */
    fp = fopen("counting.dat", "r");
    if(fp == NULL){
        printf("No counting.dat file found in your directory\n");
        exit(0);
    }
    i = 0;
    l = 0 ;
    while( i <= MAXLEV){
        if(fgets(line,sizeof(line),fp) != NULL){
            sscanf(line,"%f", &levenergy[l]);
            l++;
        }
	i++;
    }
    diml=l;//diml = l-1;
    fclose(fp);
    for(i = 0 ; i <= dim; i++){
        emin=a0+a1*((float)i - 0.5);
        emax=a0+a1*((float)i + 0.5);
        for(j = 0 ; j < diml; j++){
            if(levenergy[j]>=emin && levenergy[j]<emax){
                rholev[i]++;
            }
        }
        l = rholev[i];
        rholev[i]=rholev[i]/(fabs(a1/1000.));
    }
	
    /* ***************************************************************************** */
    /* Trying to estimate reasonable H1, H2 limits for fitting around Bn or Bp       */
    /* Taking the 1/6 part of the upper energy data points, having reasonable errors */
    /* ***************************************************************************** */
    for(i = dim; i > 5; i--){
        if (rho[i] > 10. && drho[i] > 10 && rho[i] > 1.2*drho[i]){
            H2 = i;
            H1 = H2 - (int)(((float)H2/6.)+0.5);
            break;
        }
    }
	
    /* ***************************************************************************** */
    /* Trying to estimate reasonable L1, L2 limits for fitting around Ex=0.5-1.5 MeV */
    /* Taking data points within 1 MeV, and test that there are no 0 points          */
    /* ***************************************************************************** */
    i1 = (int)(((500. - a0)/a1)+ 0.5);
    i2 = (int)((1000./a1)+0.5);
    if(i1 < 0) i1 = 1;
    if(i2 < 2) i2 = 2;
    i0 = i1;
    for(i = i1; i <= i1 + i2; i++){ /*Searching for zeros*/
        if(rholev[i] <= 0) i0 = i;
    }
    L1 = i0+1;
    L2 = L1 + i2;
    if(L2 > H1 -2) L2 = H1 - 2;
	
    /* ***************************************************************************** */
    /* Trying to estimate reasonable H1, H2 limits for fitting high T(Eg)            */
    /* Taking the 1/6 part of the upper energy data points, having reasonable errors */
    /* ***************************************************************************** */
    for(i = dim; i > dim/2; i--){
        if (sig[i] > 0. && dsig[i] > 0. && sig[i] > 1.2*dsig[i]){
            TH2 = i;
            TH1 = TH2 - (int)(((float)TH2/6.)+0.5);
            break;
        }
    }
    for(i = dim; i > dim/2; i--){
        if (sig[i] > 0. && dsig[i] > 0.){
            Tlast = i;
            break;
        }
    }
    
    /* ***************************************************************************** */
    /* Trying to estimate reasonable L1, L2 limits for fitting low T(Eg)             */
    /* Taking data points within 1 MeV, and test that there are no 0 points          */
    /* ***************************************************************************** */
    for(i = 0; i < dim/2; i++){
        if (sig[i] > 0. && dsig[i] > 0. && sig[i] > 1.2*dsig[i]){
            i1 = i;
            break;
        }
    }
    for(i = 0; i < dim/2; i++){
        if (sig[i] > 0. && dsig[i] > 0.){
            Tfirst = i;
            break;
        }
    }
	
    if(i1 > dim/2) i1 = dim/2;
    i2 = i1 + (int)((1000./a1)+0.5);
    if(i1 < 0) i1 = 1;
    if(i2 < 2) i2 = 2;
    i0 = i1;
    for(i = i1; i <= i1 + i2; i++){ /*Searching for zeros*/
        if(sig[i] <= 0) i0 = i;
    }
    TL1 = i0+1;
    TL2 = TL1 + (i2-i1);
    if(TL2 > TH1 -2) TL2 = TH1 - 2;
	
    /* ***************************************** */
    /* Reading default values from previous runs */
    /* ***************************************** */
    fp = fopen("input.cnt", "r");
    if(fp == NULL){
        printf("\nCould not open file input.cnt, default values are used \n");
    }
    else {
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %f %f %f %f %f \n", &Amass, &Delta, &Bn, &rho0, &drho0);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %d %d %d %d \n", &L1, &L2, &H1, &H2);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %d %d %d %d \n", &TL1, &TL2, &TH1, &TH2);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %d %f %f \n", &isig, &aRobin, &E1Robin);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %d %f %f \n", &itemp, &TRobin, &E0Robin);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %d  \n", &imodel);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %d %f %f \n", &ansL, &abestL, &bbestL);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %d %f %f \n", &ansH, &abestH, &bbestH);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %f \n", &red);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %f %f %f %f \n", &exe1, &sige1, &exe2, &sige2);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line, " %f \n", &part_res);
        fclose(fp);
    }
    // For the first time running
    if(abestL==-1000.)ansL = 0;
    if(abestH==-1000.)ansH = 0;
    
    /* *********************** */
    /* Asking for input values */
    /* *********************** */

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%f",&part_res);*/
    if (part_res > 0){
        ReadSmooth(part_res);
    }
    
    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%f", &Amass);*/
    
    if((int)(Amass/2.)*2==(int)Amass){ //determine if integer spin or half spin system
        start_spin=0.;
    }else{
        start_spin=0.5;
    }

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%f", &Bn);*/
    if(Bn > 100){
        printf("\nToo big number");
        printf("\nCome back another day\n\n");
        return(0);
    }

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &imodel);*/
    if(imodel < 1 || imodel > 2) exit(0);
    
    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &isig);*/
    if(isig < 1 || isig > 5) exit(0);
    
    if(isig == 1 || isig == 2){
        itemp = 1;
        if(isig==1)itemp=2;
        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%d", &itemp);*/
        if(itemp < 1 || itemp > 2) exit(0);
    }
    
    if(isig == 1){
        if(red < 0.05 || red > 1.5)red = 1.0;
        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &red);*/
        if(red < 0.05 || red > 1.5){
            printf("\nSorry, not your day today...\n");
            exit(0);
        }
    }
    
    if(isig == 5){
        if(exe2 < 0)exe2=Bn; //for the first run


        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &exe1);*/

        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &sige1);*/
        sige1 = sige1*sige1;

        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &exe2);*/

        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &sige2);*/
        sige2=sige2*sige2;
        b2 = (sige2-sige1)/(exe2-exe1);
        b1 = sige1 - b2*exe1;

        if(b2 < 0.005 || b2 > 100.){
            printf("\nCrazy spin cutoff parameters. Sorry, not your day today...\n");
            exit(0);
        }
    }
    

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%f", &rho0);*/

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%f", &drho0);*/
	
    /* *********************************************************************** */
    /* Asking or calculating Fermi-gas parameters according to case (1) or (2) */
    /* *********************************************************************** */
     if(isig == 1||isig == 2||isig == 4||imodel == 2){

        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &aRobin);*/

        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &E1Robin);*/
        a  = aRobin;
        E1 = E1Robin;
    }
    if(imodel == 1){

        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &TRobin);*/
        E0Robin = Bn - TRobin*(log(rho0)+log(TRobin));

        T  = TRobin;
        E0 = E0Robin;
    }
    
    /* *********************************************/
    /* Asking for fit limits L1, L2, H1, H2 for Rho*/
    /* *********************************************/
    i1 = 0;
    i2 = L2 + (int)((1000./a1)+0.5);


    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &L1);*/

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &L2);*/
	
    i1 = H1 - (int)((1000./a1)+0.5);
    i2 = H2 + (int)((1000./a1)+0.5);
    if(i1 < L2)   i1 = L2;
    if(i2 > dim) i2 = dim;

	

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &H1);*/

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &H2);*/
	
    /* ********************************************/
    /* Asking for fit limits L1, L2, H1, H2 for T */
    /* ********************************************/
    i1 = 0;
    i2 = TL2 + (int)((1000./a1)+0.5);


    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &TL1);*/

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &TL2);*/
	
    i1 = TH1 - (int)((1000./a1)+0.5);
    i2 = TH2 + (int)((1000./a1)+0.5);
    if(i1 < TL2)   i1 = TL2;
    if(i2 > dim) i2 = dim;
	

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &TH1);*/

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &TH2);*/
	
    /* ************************************************************* */
    /* All inputs are now in place                                   */
    /* Now calculating the Fermi gas formula for the case (1) or (2) */
    /* And determine the eta-parameter to match Rho at Bn or Bp      */
    /* ************************************************************* */
    
    rhofg(Amass, Bn, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
    eta = rho0/rhox;

    rhofg(Amass, Bn, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
	spcu = sqrt(sig2);


//////////////////////////////////////////////
    
    /* ******************************* */
    /* Searching the best T parameter  */
    /* ******************************* */
    if(imodel == 1){
        Tlow = 0.8*T;
        Tlow = Tlow*100.;
        i = (int)(Tlow + .5);
        Tlow = (float)i/100.;
        Thigh = 1.2*T;
        dTx = 0.005;
        Tx = Tlow;
        // int freeL = L2 - L1 - 1; // -1 due to one fit parameter
        int freeH = H2 - H1 - 1; // -1 due to one fit parameter
        float leastchi = 999999999.;
        
        while( Tx <= Thigh){
            E0x = Bn - Tx*(log(rho0)+log(Tx));
            searchT();
            // printf("\n T = %7.3f, E0 = %7.3f, Chi2 = %9.3f, Chi2/freeH = %9.3f",Tx, E0x, chi2_highx, chi2_highx/freeH);
            if(chi2_highx < leastchi){
            	leastchi = chi2_highx;
            	TRobin = Tx;
            }
            Tx = Tx + dTx;
        }

        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &TRobin);*/
        E0Robin = Bn - TRobin*(log(rho0)+log(TRobin));
        T  = TRobin;
        E0 = E0Robin;
    }
 

    
//////////////////////////////////////////////
    
    /* **************************** */
    /* Searching the normalization  */
    /* parameters Anorm and alpha   */
    /* **************************** */
    searchAalpha();
	
    /* ******************************************************* */
    /* Normalization parameters Anorm and alpha are determined */
    /* ******************************************************* */
	
    /* ************************************************************* */
    /* Writing Fermi-gas level density to high excitation energy     */
    /* ************************************************************* */
    FGmax = (int)(100./(a1/1000.));
    fp = fopen("fermigas.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file fermigas.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i <= FGmax; i++){
            ex = (a0+a1*(float)i)/1000.;
            rhofg(Amass, ex, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
            fprintf(fp, " %14.7e \n", eta*rhox);
        }
    }
    fclose(fp);

    /* ************************************************************* */
    /* Writing level density to high excitation energy. Above H2     */
    /* the FG or CT expression is used. Making rhoobs for TALYS      */
    /* ************************************************************* */
    fp = fopen("rhotmopaw.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file rhotmopaw.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i <= FGmax; i++){
            ex = (a0+a1*(float)i)/1000.;
            if(i < H1){
                fprintf(fp, " %14.7e \n", Anorm*exp(alpha*ex)*rho[i]); //Changed from H2 to H1, 19 Jul 2016
                rhoobs[i] = Anorm*exp(alpha*ex)*rho[i];
            }
            rhofg(Amass, ex, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
            if(i >=  H1){
                fprintf(fp, " %14.7e \n", eta*rhox);
                rhoobs[i] = eta*rhox;
            }
        }
    }
    fclose(fp);
	
    /* ****************************************** */
    /* Normalized rhopaw.cnt are written to disk  */
    /* ****************************************** */
    fp = fopen("rhopaw.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file rhopaw.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i <= dim; i++){
            ex = (a0+a1*(float)i)/1000.;
			x = Anorm*exp(alpha*ex)*rho[i];
			if(x < eps) x = eps_0;

            fprintf(fp, " %14.7e \n", x);
        }
        for (i = 0; i <= dim; i++){
            ex = (a0+a1*(float)i)/1000.;
			x = Anorm*exp(alpha*ex)*drho[i];
			if(x < eps) x = eps_0;
            fprintf(fp, " %14.7e \n", x);
        }
    }
    fclose(fp);
	
    /* ************************************************ */
    /* Normalized slope sigpaw.cnt are written to disk  */
    /* ************************************************ */
    fp = fopen("sigpaw.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file sigpaw.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i <= dim; i++){
            eg = (a0+a1*(float)i)/1000.;
            nsig[i] = Anorm*exp(alpha*eg)*sig[i];
			if(nsig[i] < eps) nsig[i] = eps_0;
            fprintf(fp, " %14.7e \n", nsig[i]);
        }
        for (i = 0; i <= dim; i++){
            eg = (a0+a1*(float)i)/1000.;
            ndsig[i] = Anorm*exp(alpha*eg)*dsig[i];
			if(ndsig[i] < eps) ndsig[i] = eps_0;
            fprintf(fp, " %14.7e \n", ndsig[i]);
        }
    }
    fclose(fp);
	

    /* ****************************************** */
    /* Extending sigpaw.cnt to Eg = 0 and Eg = Bn */
    /* Called sigextpaw.cnt                       */
    /* ****************************************** */
    sigdimBn = (((Bn + 0.5 - (a0/1000.))/(a1/1000.))+0.5);
    if(sigdimBn < dim)sigdimBn = dim;
    extendL();
    extendH();
    fp = fopen("sigextpaw.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file sigextpaw.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i < TL2+1; i++){
            fprintf(fp, " %14.7e \n", nsigL[i]);
        }
        for (i = TL2+1; i < TH1 ; i++){
            fprintf(fp, " %14.7e \n", nsig[i]);
        }
        for (i = TH1; i <= FGmax/5; i++){       // taken from lower value TH1 and upwards in gamma energy (previously from TH2) Changed 19 Jul 2016
            fprintf(fp, " %14.7e \n", nsigH[i]); // up to to 20 MeVchanged 20 Dec 2016
        }
    }
    fclose(fp);
	
    fp = fopen("extendLH.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file extendLH.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i <= FGmax/5; i++){
            fprintf(fp, " %14.7e  %14.7e \n", nsigL[i],nsigH[i] );
        }
    }
    fclose(fp);
	

    /* ********************************************* */
    /* Density from known levels are written to disk */
    /* ********************************************* */
    fp = fopen("rholev.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file rholev.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i <= dim; i++){
            eg = (a0+a1*(float)i)/1000.;
			if(rholev[i] < eps) rholev[i] = eps_0;
            fprintf(fp, " %14.7e \n", rholev[i]);
        }
    }
    fclose(fp);	

    /* ****************************************** */
    /* Spin cutoff parameters are written to disk */
    /* ****************************************** */
    fp = fopen("spincut.cnt", "w+");
    if(fp == NULL){
        printf("Could not open file spincut.cnt \n");
        exit(0);
    }
    else {
        for (i = 0; i <= 3*dim; i++){
            ex = (a0+a1*(float)i)/1000.;
            rhofg(Amass, ex, a, T, E1, E0, isig, itemp, imodel,red, b1, b2);
            fprintf(fp, " %14.7e \n", sqrt(sig2));
        }
        for (i = 0; i <= FGmax; i++){
            ex = (a0+a1*(float)i)/1000.;
            rhofg(Amass, ex, a, T, E1, E0, isig, itemp, imodel,red, b1, b2);
            sigma2[i] = sig2;
        }

    }
    fclose(fp);	
    
    /* ************************************ */
    /* NLDs on TALYS format written to disk */
    /* ************************************ */
    nld_talys();
    
    
    

    /* ***************************************************** */
    /* Reading matrix fg.rsg with calibration and dimensions WHY DO I DO ALL THIS??? OLD STUFF?? */
    /* ***************************************************** */
/*    printf("Reading file fg.rsg\n");
    fp = fopen("fg.rsg", "r");
    if(fp == NULL){
        printf("No fg.rsg file found in your directory\n");
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
        sscanf(line,"%13s %7s %f %1s %f %1s %f %1s %f %1s %f %1s %f",cdum,cdum,&a0_x,cdum,&a1_x,cdum,&a2_x,cdum,&a0_y,cdum,&a1_y,cdum,&a2_y);
        fgets_ignore(line,sizeof(line),fp);
        fgets_ignore(line,sizeof(line),fp);
        sscanf(line,"%s %d %3s %d",cdum, &dim_x, cdum, &dim_y);
        fgets_ignore(line,sizeof(line),fp);
    }
    printf("Matrix fg.rsg has dimension (0 : %d, 0 : %d) and calibrations: \n",dim_x,dim_y);
    printf("x-axis: (a0, a1, a2) = (%f, %f, %f)\n",a0_x,a1_x,a2_x);
    printf("y-axis: (a0, a1, a2) = (%f, %f, %f)\n",a0_y,a1_y,a2_y);

    for(j = 0; j <= dim_y; j++){
        for(i = 0; i <= dim_x; i++){
            fscanf(fp,"%f", &fg[i][j]);
        }
    }
    fclose(fp);
    
*/
    /* **************************************************** */
    /* Storing default values for the next run in input.cnt */
    /* **************************************************** */
    fp = fopen("input.cnt", "w");
    if(fp == NULL){
        printf("Could not open file input.cnt \n");
        exit(0);
    }
    else {
        fprintf(fp, " %f %f %f %f %f \n", Amass, Delta, Bn, rho0, drho0);
        fprintf(fp, " %d %d %d %d \n", L1, L2, H1, H2);
        fprintf(fp, " %d %d %d %d \n", TL1, TL2, TH1, TH2);
        fprintf(fp, " %d %f %f \n", isig, aRobin, E1Robin);
        fprintf(fp, " %d %f %f \n", itemp, TRobin, E0Robin);
        fprintf(fp, " %d \n", imodel);
        fprintf(fp, " %d %f %f \n", ansL, abestL, bbestL);
        fprintf(fp, " %d %f %f \n", ansH, abestH, bbestH);
        fprintf(fp, " %f \n",red);
        fprintf(fp, " %f %f %f %f \n", exe1, sqrt(sige1), exe2, sqrt(sige2));
        fprintf(fp, " %f \n", part_res);
    }
    fclose(fp);
    
    makeroot1();
    makeroot2();
    makeroot3();
    return(0);
}

void rhofg(float Amass, float ex, float a, float T, float E1, float E0, int isig, int itemp, int imodel, float red, float b1, float b2){
    float exx, uCT=0.005, uFG=0.005, vv;
    rhox = 0.1;
    exx = ex;
    if(exx <= 0.005)exx = 0.005;
    uCT = exx - E0;
    uFG = exx - E1;
    if(uCT < 0.005) uCT = 0.005;
    if(uFG < 0.005) uFG = 0.005;
    
    if(isig == 1){
        if(itemp == 1) sig2 = red*0.0146*pow(Amass,(5./3.))*sqrt(uFG/a);
        if(itemp == 2) sig2 = red*0.0146*pow(Amass,(5./3.))*(1. + sqrt(1. + 4.*a*uFG))/(2.*a);
    }
    if(isig == 2){
        if(itemp == 1) sig2 = 0.0888*pow(Amass,(2./3.))*a*sqrt(uFG/a);
        if(itemp == 2) sig2 = 0.0888*pow(Amass,(2./3.))*a*(1. + sqrt(1. + 4.*a*uFG))/(2.*a);
    }
    if(isig == 3) {                                         /* Constant temperature formula */
        sig2 = 0.98*pow(Amass,0.29)*0.98*pow(Amass,0.29);   /* Does not depend on Ex */
    }
    if(isig == 4){
        vv   = ex - (E1+0.381);     /* vv is (Ex-0.5Pa_prime)**0.381, with 0.5Pa_prime=E1+0.381 */
        if(vv < 0.0050)vv = 0.0050;
        sig2 = 0.391 *pow(Amass,(0.675))*pow(vv,0.312);
    }
    if(isig == 5){
        sig2 = b1 + b2*ex;
    }
    
    if(sig2 < 0.5) sig2 = 0.5;
    if(imodel == 1)rhox = (1./T)*exp(uCT/T);
    if(imodel == 2){
        if(uFG < (25./16.)/a) uFG = (25./16.)/a;
        rhox = exp(2.*sqrt(a*uFG))/(12.*sqrt(2.*sig2)*pow(a,(1./4.))*pow(uFG,(5./4.)));
    }
    return ;
}


int makeroot1(){
    float Emin, Emax, half, exx, ex0, lower, eL1, eL2, eH1, eH2, cL1, cL2, cH1, cH2, Rhomin, Rhomax;
    int Hmin, Hmax, dH;
    Emax = Bn + 0.5;
    Hmax = (((Bn*1000. + 500. - a0)/a1)+0.5);
    Emax = (a0 + a1 * Hmax)/1000.;
    Emin = a0/1000.;
    if(Emin > 0.)Emin = 0.;
    Hmin = H1 - (H2 - H1);
    if(Hmin <  Hmax/2.)Hmin =Hmax/2.;
    if(Hmin > H1) Hmin = H1;
    dH   = Hmax - Hmin +1;
    Rhomin = 1000000.;
    for (i = 0; i <= dim; i++){
        ex = (a0+a1*(float)i)/1000.;		
        if (Anorm*exp(alpha*ex)*rho[i] < Rhomin && Anorm*exp(alpha*ex)*rho[i] > 0.0001 ) Rhomin = 0.9*Anorm*exp(alpha*ex)*rho[i];
    }
    if (Rhomin < 0.0001) Rhomin = 0.0001;
    exx = log10(10.*rho0) + log10(Rhomin);
    ex0 = log10(10.*rho0);
    half = pow(10.,exx/2.);
    lower = Rhomin*0.3;
    eL1 = (a0 + a1 * L1)/1000.;
    eL2 = (a0 + a1 * L2)/1000.;
    eH1 = (a0 + a1 * H1)/1000.;
    eH2 = (a0 + a1 * H2)/1000.;
    cL1 = Anorm*exp(alpha*eL1)*rho[L1]*pow(10.,ex0/10.);
    cL2 = Anorm*exp(alpha*eL2)*rho[L2]*pow(10.,ex0/10.);
    cH1 = Anorm*exp(alpha*eH1)*rho[H1]*pow(10.,ex0/10.);
    cH2 = Anorm*exp(alpha*eH2)*rho[H2]*pow(10.,ex0/10.);
    Rhomax = 2.*cH2*pow(10.,exx/10.);
    if(Rhomax < 10.*rho0) Rhomax = 10.*rho0;
    fp = fopen("counting.cpp", "w+");
    if(fp == NULL){
        printf("Could not open file counting.cpp \n");
        exit(0);
    }
    else {
        fprintf(fp,"{\n");
        fprintf(fp,"   gROOT->Reset();\n");
        fprintf(fp,"   gROOT->SetStyle(\"Plain\");\n");
        fprintf(fp,"   gStyle->SetOptTitle(0);\n");
        fprintf(fp,"   gStyle->SetOptStat(0);\n");
        fprintf(fp,"   gStyle->SetFillColor(0);\n");
        fprintf(fp,"   gStyle->SetPadBorderMode(0);\n");
        fprintf(fp,"   m = (TH1F*)gROOT->FindObject(\"h\");\n");
        fprintf(fp,"   if (m) m->Delete();\n");
        fprintf(fp,"   TCanvas *c1 = new TCanvas(\"c1\",\"Normalization of level density\",600,600);\n");
        fprintf(fp,"   TH2F *h = new TH2F(\"h\",\" \",10,%f,%f,50,%f,%f);\n",Emin,Emax,Rhomin,Rhomax);
        fprintf(fp,"   ifstream rholev(\"rholev.cnt\"), rhopaw(\"rhopaw.cnt\"), fermi(\"fermigas.cnt\");\n");
        fprintf(fp,"   float levels[%d],rho[%d],rhoerr[%d],energy[%d],energyerr[%d],fermigas[%d];\n",dim+2,dim+2,dim+2,FGmax+2,FGmax+2,FGmax+2);
        fprintf(fp,"   float Bn[1]={%f};\n",Bn);
        fprintf(fp,"   float Bnerr[1]={0.001};\n");
        fprintf(fp,"   float rho_Bn[1]={%f};\n",rho0);
        fprintf(fp,"   float rho_Bnerr[1]={%f};\n",drho0);
        fprintf(fp,"   int i = 0;\n");
        fprintf(fp,"   float a0 = %8.4f;\n",a0/1000.);
        fprintf(fp,"   float a1 = %8.4f;\n",a1/1000.);
        fprintf(fp,"   float x,y,z;\n");
        fprintf(fp,"   while(fermi){\n");
        fprintf(fp,"   	fermi >> x;\n");
        fprintf(fp,"   	fermigas[i]=x;\n");
        fprintf(fp,"   	energy[i]=a0+(a1*i);\n");
        fprintf(fp,"   	energyerr[i]=0.0;\n");
        fprintf(fp,"      i++;\n");
        fprintf(fp,"   }\n");
        fprintf(fp,"   i=0;\n");
        fprintf(fp,"   while(rhopaw){\n");
        fprintf(fp,"   	rhopaw >> y;\n");
        fprintf(fp,"   	if(i<%d){\n",dim+1);
        fprintf(fp,"   		rho[i]=y;\n");
        fprintf(fp,"   	}\n");
        fprintf(fp,"   	else{rhoerr[i-%d]=y;}\n",dim+1);
        fprintf(fp,"   	i++;\n");
        fprintf(fp,"   }\n");
        fprintf(fp,"  	i=0;\n");
        fprintf(fp,"	while(rholev){\n");
        fprintf(fp,"		rholev >> z;\n");
        fprintf(fp,"		levels[i]=z;\n");
        fprintf(fp,"		i++;\n");
        fprintf(fp,"  }\n");
        fprintf(fp,"   TGraphErrors *rhoexp = new TGraphErrors(%d,energy,rho,energyerr,rhoerr);\n",dim+1);
        fprintf(fp,"   TGraphErrors *rhoBn = new TGraphErrors(1,Bn,rho_Bn,Bnerr,rho_Bnerr);\n");
        fprintf(fp,"   TGraph *fermicalc = new TGraph(%d,energy,fermigas);\n",FGmax+1);
        fprintf(fp,"   TGraph *level = new TGraph(%d,energy,levels);\n",dim+1);
        fprintf(fp,"   c1->SetLogy();\n");
        fprintf(fp,"   c1->SetLeftMargin(0.14);\n");
        fprintf(fp,"   h->GetXaxis()->CenterTitle();\n");
        fprintf(fp,"   h->GetXaxis()->SetTitle(\"Excitation energy E (MeV)\");\n");
        fprintf(fp,"   h->GetYaxis()->CenterTitle();\n");
        fprintf(fp,"   h->GetYaxis()->SetTitleOffset(1.4);\n");
        fprintf(fp,"   h->GetYaxis()->SetTitle(\"Level density #rho (E) (MeV^{-1})\");\n");
        fprintf(fp,"   h->Draw();\n");
        fprintf(fp,"   rhoexp->SetMarkerStyle(21);");
        fprintf(fp,"   rhoexp->SetMarkerSize(0.8);\n");
        fprintf(fp,"   rhoexp->Draw(\"P\");\n");
        fprintf(fp,"   fermicalc->SetLineStyle(2);\n");
        fprintf(fp,"   fermicalc->DrawGraph(%d,&fermicalc->GetX()[%d],&fermicalc->GetY()[%d],\"L\");\n",dH,Hmin,Hmin);
        fprintf(fp,"   level->SetLineStyle(1);\n");
        fprintf(fp,"   level->Draw(\"L\");\n");
        fprintf(fp,"   rhoBn->SetMarkerStyle(25);\n");
        fprintf(fp,"   rhoBn->SetMarkerSize(0.8);\n");
        fprintf(fp,"   rhoBn->Draw(\"P\");\n");
        fprintf(fp,"   TLegend *leg = new TLegend(0.15,0.70,0.6,0.85);\n");
        fprintf(fp,"   leg->SetBorderSize(0);\n");
        fprintf(fp,"   leg->SetFillColor(0);\n");
        fprintf(fp,"   leg->AddEntry(rhoexp,\" Oslo data \",\"P\");\n");
        fprintf(fp,"   leg->AddEntry(level,\" Known levels \",\"L\");\n");
        fprintf(fp,"   leg->AddEntry(fermicalc,\" CT or FG model \",\"L\");	\n");
        fprintf(fp,"   leg->AddEntry(rhoBn,\" #rho from neutron res. data \",\"P\");\n");
        fprintf(fp,"   leg->Draw();\n");
        fprintf(fp,"   TLatex t;\n");
        fprintf(fp,"   t.SetTextSize(0.05);\n");
        fprintf(fp,"   t.DrawLatex(%9.3f,%9.3e,\"^{xx}Yy\");\n",0.8*Emax,0.5*Rhomax);
        fprintf(fp,"   TArrow *arrow1 = new TArrow(%f,%f,%f,%f,0.02,\">\");\n",eL1,cL1*pow(10.,ex0/10.),eL1,cL1);
        fprintf(fp,"   arrow1->Draw();\n");
        fprintf(fp,"   TArrow *arrow2 = new TArrow(%f,%f,%f,%f,0.02,\">\");\n",eL2,cL2*pow(10.,ex0/10.),eL2,cL2);
        fprintf(fp,"   arrow2->Draw();\n");
        fprintf(fp,"   TArrow *arrow3 = new TArrow(%f,%f,%f,%f,0.02,\">\");\n",eH1,cH1*pow(10.,ex0/10.),eH1,cH1);
        fprintf(fp,"   arrow3->Draw();\n");
        fprintf(fp,"   TArrow *arrow4 = new TArrow(%f,%f,%f,%f,0.02,\">\");\n",eH2,cH2*pow(10.,ex0/10.),eH2,cH2);
        fprintf(fp,"   arrow4->Draw();\n");
        fprintf(fp,"   c1->Update();\n");
		fprintf(fp,"   c1->Print(\"counting.pdf\");\n");
        fprintf(fp,"   c1->Print(\"counting.eps\");\n");
        fprintf(fp,"   c1->Print(\"counting.ps\");\n");
        fprintf(fp,"}\n");
    }
    fclose(fp);
    return 0;
}

int makeroot2(){
    int dimcut;
    float emin,emax;
    float cmax;
    dimcut = 3*dim;
    emax = (a0 + a1*(float)dimcut)/1000.;
    emin = a0/1000.;
    if(emin > 0.)emin = 0.;
    rhofg(Amass, ex, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
    cmax = sqrt(sig2);
    fp = fopen("spincut.cpp", "w+");
    if(fp == NULL){
        printf("Could not open file spincut.cpp \n");
        exit(0);
    }
    else {
        fprintf(fp,"{\n");
        fprintf(fp,"	gROOT->Reset();\n");
        fprintf(fp,"	gROOT->SetStyle(\"Plain\");\n");
        fprintf(fp,"   gStyle->SetOptTitle(0);\n");	
        fprintf(fp,"   gStyle->SetOptStat(0);\n");
        fprintf(fp,"   gStyle->SetFillColor(0);\n");
        fprintf(fp,"   gStyle->SetPadBorderMode(0);\n");
        fprintf(fp,"   m = (TH1F*)gROOT->FindObject(\"h\");\n"); 
        fprintf(fp,"   if (m) m->Delete();\n"); 
        fprintf(fp,"   TCanvas *c1 = new TCanvas(\"c1\",\"Spincut parameter\",600,600);\n");	
        fprintf(fp,"   TH2F *h = new TH2F(\"h\",\" \",10,%f,%f,50,0.001,%7.1f);\n",emin,emax,cmax+1.);
        fprintf(fp,"   ifstream spincutfile(\"spincut.cnt\");\n");
        fprintf(fp,"   float energy[%d],spincut[%d];\n",dimcut+2,dimcut+2);
        fprintf(fp,"   int i = 0;\n");
        fprintf(fp,"   float a0 = %8.4f;\n",a0/1000.); 
        fprintf(fp,"   float a1 = %8.4f;\n",a1/1000.);
        fprintf(fp,"   float x,y,z;\n");
        fprintf(fp,"   while(spincutfile){\n");
        fprintf(fp,"   	spincutfile >> x;\n");
        fprintf(fp,"	   spincut[i]=x;\n");
        fprintf(fp,"	   energy[i]=a0+(a1*i);\n");
        fprintf(fp,"	   i++;\n");
        fprintf(fp,"   }\n");
        fprintf(fp,"   TGraph *spincutgraph = new TGraph(%d,energy,spincut);\n",dimcut+1);
        fprintf(fp,"   c1->SetLeftMargin(0.14);\n");
        fprintf(fp,"   h->GetXaxis()->CenterTitle();\n");
        fprintf(fp,"   h->GetXaxis()->SetTitle(\"Excitation energy E (MeV)\");\n");
        fprintf(fp,"   h->GetYaxis()->CenterTitle();\n");
        fprintf(fp,"   h->GetYaxis()->SetTitleOffset(1.4);\n");
        fprintf(fp,"   h->GetYaxis()->SetTitle(\"Spin cutoff #sigma\");\n");
        fprintf(fp,"   h->Draw();\n");
        fprintf(fp,"   spincutgraph->Draw(\"L\");\n");
        fprintf(fp,"   c1->Update();\n");
		fprintf(fp,"   c1->Print(\"spincut.pdf\");\n");
        fprintf(fp,"   c1->Print(\"spincut.eps\");\n");
        fprintf(fp,"   c1->Print(\"spincut.ps\");\n");
        fprintf(fp,"}\n");
    }
    fclose(fp);
    return 0;
}

int makeroot3(){
    float Emin, Emax, half, exx, ex0, lower, eL1, eL2, eH1, eH2, cL1, cL2, cH1, cH2, T0, Tmin, Tmax;
    int Hmin, Hmax;
    //    Emax = Bn + 0.5;
    Hmax = (((Bn + 1.5 - (a0/1000.))/(a1/1000.))+0.5); // 1500 above the Bn
    //    Emax = (a0/1000.) + (a1/1000.) * Hmax;
    Emax = 1.0 + (a0/1000.) + (a1/1000.) * dim ; // 1 MeV above the most upper data point
    
    if(sigdimBn < dim)sigdimBn=dim;
    
    Emin = a0/1000.;
    if(sigdimBn > Hmax)Hmax=sigdimBn;
    if(Emin > 0.)Emin = 0.;
    Hmin = TH1 - (TH2 - TH1);
    if(Hmin <  Hmax/2.)Hmin =Hmax/2.;
    if(Hmin > TH1) Hmin = TH1;
    Tmin = 1000000.;
    for (i = 0; i <= dim; i++){
        eg = (a0+a1*(float)i)/1000.;
        if (Anorm*exp(alpha*eg)*sig[i] < Tmin && Anorm*exp(alpha*eg)*sig[i] > 0.0001 ) Tmin = 0.1*Anorm*exp(alpha*eg)*sig[i];
    }
    if (Tmin < 0.0001) Tmin = 0.0001;
    T0=0.0001;
    for (i = 0; i <= dim; i++){
        eg = (a0+a1*(float)i)/1000.;
        if (Anorm*exp(alpha*eg)*sig[i] > T0 && Anorm*exp(alpha*eg)*sig[i] > 0.0001 ) T0 = 10.*Anorm*exp(alpha*eg)*sig[i];
    }
    exx = log10(10.*T0) + log10(Tmin);
    ex0 = log10(10.*T0);
    half = pow(10.,exx/2.);
    lower = Tmin*0.3;
    eL1 = (a0 + a1 * TL1)/1000.;
    eL2 = (a0 + a1 * TL2)/1000.;
    eH1 = (a0 + a1 * TH1)/1000.;
    eH2 = (a0 + a1 * TH2)/1000.;
    cL1 = Anorm*exp(alpha*eL1)*sig[TL1]*pow(10.,ex0/10.);
    cL2 = Anorm*exp(alpha*eL2)*sig[TL2]*pow(10.,ex0/10.);
    cH1 = Anorm*exp(alpha*eH1)*sig[TH1]*pow(10.,ex0/10.);
    cH2 = Anorm*exp(alpha*eH2)*sig[TH2]*pow(10.,ex0/10.);
    Tmax = 2.*cH2*pow(10.,exx/10.);
    if(Tmax < 10.*T0) Tmax = 10.*T0;

    fp = fopen("sigext.cpp", "w+");
    if(fp == NULL){
        printf("Could not open file sigext.cpp \n");
        exit(0);
    }
    else {
	fprintf(fp,"{\n");
	fprintf(fp,"   gROOT->Reset();\n");
	fprintf(fp,"   gROOT->SetStyle(\"Plain\");\n");
	fprintf(fp,"   gStyle->SetOptTitle(0);\n");
	fprintf(fp,"   gStyle->SetOptStat(0);\n");
	fprintf(fp,"   gStyle->SetFillColor(0);\n");
	fprintf(fp,"   gStyle->SetPadBorderMode(0);\n");
	fprintf(fp,"   m = (TH1F*)gROOT->FindObject(\"h\");\n"); 
	fprintf(fp,"   if (m) m->Delete();\n");
	fprintf(fp,"   TCanvas *c1 = new TCanvas(\"c1\",\"Normalization of gamma-transmission coefficient\",600,600);\n");	
	fprintf(fp,"   TH2F *h = new TH2F(\"h\",\" \",10,%f,%8.3f,50,%9.3e,%9.3e);\n",Emin,Emax,Tmin,Tmax);
	fprintf(fp,"   ifstream sigfile(\"sigpaw.cnt\");\n");
	fprintf(fp,"   float sig[%d],sigerr[%d];\n",dim+2,dim+2);
        fprintf(fp,"   float energy[%d],energyerr[%d];\n",(FGmax/5)+1,(FGmax/5)+1);
        fprintf(fp,"   float extL[%d],extH[%d];\n",(FGmax/5)+2,(FGmax/5)+2);
	fprintf(fp,"   int i;\n");
	fprintf(fp,"   float a0 =%8.4f;\n",a0/1000.);
	fprintf(fp,"   float a1 =%8.4f;\n",a1/1000.);
        fprintf(fp,"   for(i = 0; i < %d; i++){\n",sigdimBn+1);
        fprintf(fp,"   	energy[i] = a0 + (a1*i);\n");
        fprintf(fp,"   	energyerr[i] = 0.0;\n");
        fprintf(fp,"   	extL[i] = 0.0;\n");
        fprintf(fp,"   	extH[i] = 0.0;\n");
        fprintf(fp,"   }\n");
        fprintf(fp,"   float x, y;\n");
        fprintf(fp,"   i = 0;\n");
        fprintf(fp,"   while(sigfile){\n");
        fprintf(fp,"   	sigfile >> x;\n");
        fprintf(fp,"   	if(i<%d){\n",dim+1);
        fprintf(fp,"   		sig[i]=x;\n");
        fprintf(fp,"   	}\n");
        fprintf(fp,"   	else{sigerr[i-%d]=x;}\n",dim+1);
        fprintf(fp,"   	i++;\n");
        fprintf(fp,"   }\n");
        fprintf(fp,"   ifstream extendfile(\"extendLH.cnt\");\n");
        fprintf(fp,"   i = 0;\n");
        fprintf(fp,"   while(extendfile){\n");
        fprintf(fp,"   	extendfile >> x >> y ;\n");
        fprintf(fp,"   	extL[i]=x;\n");
        fprintf(fp,"   	extH[i]=y;\n");
        fprintf(fp,"   	i++;\n");
        fprintf(fp,"   }\n");
        fprintf(fp,"   TGraph *extLgraph = new TGraph(%d,energy,extL);\n",sigdimBn+1);
        fprintf(fp,"   TGraph *extHgraph = new TGraph(%d,energy,extH);\n",sigdimBn+1);
        fprintf(fp,"   TGraphErrors *sigexp = new TGraphErrors(%d,energy,sig,energyerr,sigerr);\n",dim+1);
        fprintf(fp,"   c1->SetLogy();\n");
        fprintf(fp,"   c1->SetLeftMargin(0.14);\n");
        fprintf(fp,"   h->GetXaxis()->CenterTitle();\n");
        fprintf(fp,"   h->GetXaxis()->SetTitle(\"#gamma-ray energy E_{#gamma} (MeV)\");\n");
        fprintf(fp,"   h->GetYaxis()->CenterTitle();\n");
        fprintf(fp,"   h->GetYaxis()->SetTitleOffset(1.4);\n");
        fprintf(fp,"   h->GetYaxis()->SetTitle(\"Transmission coeff. (arb. units)\");\n");
        fprintf(fp,"   h->Draw();\n");
	fprintf(fp,"   sigexp->SetMarkerStyle(21);\n");
	fprintf(fp,"   sigexp->SetMarkerSize(0.8);\n");
	fprintf(fp,"   sigexp->Draw(\"P\");\n");
        fprintf(fp,"   extLgraph->SetLineStyle(1);\n");
        fprintf(fp,"   extLgraph->DrawGraph(%d,&extLgraph->GetX()[0],&extLgraph->GetY()[0],\"L\");\n", TL2+1);
        fprintf(fp,"   extHgraph->SetLineStyle(1);\n");
        fprintf(fp,"   extHgraph->DrawGraph(%d,&extHgraph->GetX()[%d],&extHgraph->GetY()[%d],\"L\");\n",sigdimBn-TH1+1,TH1,TH1);
	fprintf(fp,"   TArrow *arrow1 = new TArrow(%9.3e,%9.3e,%9.3e,%9.3e,0.02,\">\");\n",eL1,cL1*pow(10.,ex0/10.),eL1,cL1); 
	fprintf(fp,"   arrow1->Draw();\n");
	fprintf(fp,"   TArrow *arrow2 = new TArrow(%9.3e,%9.3e,%9.3e,%9.3e,0.02,\">\");\n",eL2,cL2*pow(10.,ex0/10.),eL2,cL2);
	fprintf(fp,"   arrow2->Draw();\n");
	fprintf(fp,"   TArrow *arrow3 = new TArrow(%9.3e,%9.3e,%9.3e,%9.3e,0.02,\">\");\n",eH1,cH1*pow(10.,ex0/10.),eH1,cH1); 
	fprintf(fp,"   arrow3->Draw();\n");
	fprintf(fp,"   TArrow *arrow4 = new TArrow(%9.3e,%9.3e,%9.3e,%9.3e,0.02,\">\");\n",eH2,cH2*pow(10.,ex0/10.),eH2,cH2);
	fprintf(fp,"   arrow4->Draw();\n");
	fprintf(fp,"   c1->Update();\n");
	fprintf(fp,"   c1->Print(\"sigext.pdf\");\n");
	fprintf(fp,"   c1->Print(\"sigext.eps\");\n");
	fprintf(fp,"   c1->Print(\"sigext.ps\");\n");
	fprintf(fp,"}\n");
    }
    fclose(fp);
    
    fp = fopen("alpha.txt", "w+");
    if(fp == NULL){
        printf("Could not open file alpha.txt \n");
        exit(0);
    }
    else {
	fprintf(fp,"A = %7.4f and alpha = %6.4f\n",Anorm,alpha);
    }
    fclose(fp);
    
    return 0;
}

int searchAalpha(){
    float rhoL;
    /* ************************************************ */
    /* Take three channels in the middle of each region */
    /* in order to make a first estimate of A and alpha */
    /* ************************************************ */
    Lm = (L1 + L2)/2;
    Hm = (H1 + H2)/2;
    c1 = (rho[Lm-1]+rho[Lm]+rho[Lm+1])/3.;
    c2 = (rho[Hm-1]+rho[Hm]+rho[Hm+1])/3.;
    rhoL = (rholev[Lm-1]+rholev[Lm]+rholev[Lm+1])/3.;
    if(rhoL < 0.01) rhoL = 0.01;
    e1 = (a0+a1*(float)Lm)/1000.;
    e2 = (a0+a1*(float)Hm)/1000.;
    /*	printf("Lm=%d,Hm=%d,c1=%f,c2=%f,e1=%f,e2=%f\n",Lm,Hm,c1,c2,e1,e2);*/
    rhofg(Amass, e2, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
    alpha = (log(rhoL)+log(c2)-log(eta*rhox)-log(c1))/(e1-e2);;
    Anorm = exp(-alpha*e1)*(rhoL)/c1;
	
    /* ******************************************* */
    /* Determine c1 and c2 by weighting all points */
    /* in each region with uncertainties and slope */
    /* ******************************************* */
    c1 = c1/corrL();
    c2 = c2/corrH();
    /*	printf("Lm=%d,Hm=%d,c1=%f,c2=%f,e1=%f,e2=%f\n",Lm,Hm,c1,c2,e1,e2);*/
    rhofg(Amass, e2, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
    alpha = (log(rhoL)+log(c2)-log(eta*rhox)-log(c1))/(e1-e2);
    Anorm = exp(-alpha*e1)*(rhoL)/c1;
    return 0;
}

float corrL(){
    float corr, corrbest=0., sum, sumbest, sum0=0., cc, free, dc2, rhoL;
    corr = 0.25;
    sumbest = 1.0e+21;
    free = (float)L2 - (float)L1;
    if (free <= 0)free = 1.; 
    for(j = 0; j <= 3750; j++){
        corr = corr + 0.001;
        sum  = 0.0;
        for(i = L1; i <= L2; i++){
            ex = (a0+a1*(float)i)/1000.;
            rhoL = rholev[i];
            if(rhoL < 0.01) rhoL = 0.01;
            cc = corr*Anorm*exp(alpha*ex)*rho[i];
            dc2= (corr*Anorm*exp(alpha*ex)*drho[i])*(corr*Anorm*exp(alpha*ex)*drho[i]);
            dc2 =sqrt(dc2*dc2 + 1.*1.);
            if(dc2 > 0) sum = sum + ((cc-rhoL)*(cc-rhoL)/dc2);
        }
        sum = sum/free;
        if(j == 499){
            sum0 = sum;
        }
        if(sum <= sumbest){
            sumbest = sum;
            corrbest= corr;
        }
    }
    return corrbest;
}

float corrH(){
    float corr, corrbest=0., sum, sumbest, sum0=0., cc, free, dc2;
    corr = 0.25;
    sumbest = 1.0e+21;
    free = (float)H2 - (float)H1;
    if (free <= 0)free = 1.; 
    for(j = 0; j <= 3750; j++){
        corr = corr + 0.001;
        sum  = 0.0;
        for(i = H1; i <= H2; i++){
            ex = (a0+a1*(float)i)/1000.;
            cc = corr*Anorm*exp(alpha*ex)*rho[i];
            dc2= (corr*Anorm*exp(alpha*ex)*drho[i])*(corr*Anorm*exp(alpha*ex)*drho[i]);
            dc2 =sqrt(dc2*dc2 + 1.*1.);
            rhofg(Amass, ex, a, T, E1, E0, isig, itemp, imodel, red, b1, b2);
            if(dc2 > 0) sum=sum+(cc-eta*rhox)*(cc-eta*rhox)/dc2;
        } 
        sum = sum/free;
        if(j == 499){
            sum0 = sum;
        }
        if(sum <= sumbest){
            sumbest = sum;
            corrbest= corr;
        }
    }
    return corrbest;
}

int extendL()
{
    int i, j, k, steps = 1000;
    float x, y, yi, dyi, ai, bi, x1, x2, y1, y2;
    float c,d,d_suggest;
    float chi, aa, bb, al, bl, ah, bh, astep, bstep, chibest, abest=0., bbest=0.;
    x1 = (a0+a1*(float)TL1)/1000.;
    x2 = (a0+a1*(float)TL2)/1000.;
    y1 = log(nsig[TL1]);
    y2 = log(nsig[TL2]);
    if(TL2-TL1 > 3){
        x1 = (a0+a1*(float)TL1 + 0.5)/1000.;
        x2 = (a0+a1*(float)TL2 - 0.5)/1000.;
        y1 = (log(nsig[TL1]) + log(nsig[TL1+1]))/2.;
        y2 = (log(nsig[TL2]) + log(nsig[TL2-1]))/2.;
    }
    ai = (y2-y1)/(x2-x1);
    bi = y1 - ai*x1;
    al = ai/3.;
    ah = ai*3.;
    astep = (ah-al)/(float)steps;
    bh = bi + 2.*ai*(x2-x1);
    bl = bi - 2.*ai*(x2-x1);
    bstep = (bh-bl)/(float)steps;
//    printf(" Lower fit to sig: a = %f<%f<%f  b = %f<%f<%f\n",al,ai,ah,bl,bi,bh);
	
    chibest = 999999999.;
    bb = bl;
    for(i = 0; i < steps; i++){
        bb = bb + bstep;
        aa = al;
        for(j = 0; j < steps; j++){
            aa = aa + astep;
            chi = 0;
            for(k = TL1; k <= TL2 && nsig[k] > 0 && ndsig[k] > 0; k++){
                x   = (a0+a1*(float)k)/1000.;
                y   = aa * x + bb;
                yi  = log(nsig[k]);
                dyi = ndsig[k]/nsig[k];
                chi = chi + (y-yi)*(y-yi)/(dyi*dyi);
            }
            chi = chi/(float)(TL2-TL1);
            if(chi < chibest && chi > 0){
                chibest = chi;
                abest   = aa;
                bbest   = bb;
                /*				printf(" chi = %f a = %f b = %f \n", chi, aa, bb);  */
            }
        }
    }
    
    c = abest -(3./x1);
    d = bbest + 3.*(1.-log(x1));
    

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &ansL);*/
    
    if(ansL == 1){
        if(abestL == -1000.)abestL = c;
        if(bbestL == -1000.)bbestL = d;
        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &abestL);*/
        d_suggest = abest*x1+bbest-abestL*x1-3.*log(x1);
        fgets_ignore(line,sizeof(line),stdin);
/*        sscanf(line,"%f", &bbestL);*/

    }else{
        abestL  = c;
        bbestL  = d;
    }
    for (i = 0; i <= sigdimBn; i++){
        x = (a0+a1*(float)i)/1000.;
        nsigL[i] = exp(abest * x + bbest);
// Making the extension to low excitation as exp fuction
// This means that we should multiply with Eg**3 from the last lower experimental point
// Corrected March 31 2017
        
        
        if (i < TL1){
            nsigL[i] = x*x*x*exp(abestL * x + bbestL);
        }
        
        /*		printf("i = %d  x = %f  nsigL = %14.7e \n", i, x, nsigL[i]);*/
    }
    return 0;
}

int extendH()
{
    int i, j, k, steps = 1000;
    float x, y, yi, dyi, ai, bi, x1, x2, y1, y2;
    float chi, aa, bb, al, bl, ah, bh, astep, bstep, chibest, abest=0., bbest=0.;
    x1 = (a0+a1*(float)TH1)/1000.;
    x2 = (a0+a1*(float)TH2)/1000.;
    y1 = log(nsig[TH1]);
    y2 = log(nsig[TH2]);
    if(TH2-TH1 > 3){
        x1 = (a0+a1*(float)TH1 + 0.5)/1000.;
        x2 = (a0+a1*(float)TH2 - 0.5)/1000.;
        y1 = (log(nsig[TH1]) + log(nsig[TH1+1]))/2.;
        y2 = (log(nsig[TH2]) + log(nsig[TH2-1]))/2.;
    }
    ai = (y2-y1)/(x2-x1);
    bi = y1 - ai*x1;
    al = ai/3.;
    ah = ai*3.;
    astep = (ah-al)/(float)steps;
    bh = bi + 2.*ai*(x2-x1);
    bl = bi - 2.*ai*(x2-x1);
    bstep = (bh-bl)/(float)steps;
	
    chibest = 999999999.;
    bb = bl; 
    for(i = 0; i < steps; i++){
        bb = bb + bstep;
        aa = al;
        for(j = 0; j < steps; j++){
            aa = aa + astep;
            chi = 0;
            for(k = TH1; k <= TH2 && nsig[k] > 0 && ndsig[k] > 0; k++){
                x   = (a0+a1*(float)k)/1000.;
                y   = aa * x + bb;
                yi  = log(nsig[k]);
                dyi = ndsig[k]/nsig[k];
                chi = chi + (y-yi)*(y-yi)/(dyi*dyi);
            }
            chi = chi/(float)(TH2-TH1);
            if(chi < chibest && chi > 0){
                chibest = chi;
                abest   = aa;
                bbest   = bb;
            }
        }
    }
    

    fgets_ignore(line,sizeof(line),stdin);
/*    sscanf(line,"%d", &ansH);*/
    
    if(ansH == 1){
        if(abestH == -1000.)abestH = abest;
        if(bbestH == -1000.)bbestH = bbest;

        fgets_ignore(line,sizeof(line),stdin);
        //sscanf(line,"%f", &abestH);

        fgets_ignore(line,sizeof(line),stdin);
        //sscanf(line,"%f", &bbestH);
    }else{
        abestH  = abest;
        bbestH  = bbest;
    }
    for (i = 0; i < FGmax/5; i++){
        x = (a0+a1*(float)i)/1000.;
        nsigH[i] = exp(abestH * x + bbestH);
    }
    return 0;
}

//////////////////////////////////////////////////////////////////////////

int searchT(){
    float rhoL;
    /* ************************************************ */
    /* Take three channels in the middle of each region */
    /* in order to make a first estimate of A and alpha */
    /* ************************************************ */
    Lm = (L1 + L2)/2;
    Hm = (H1 + H2)/2;
    c1 = (rho[Lm-1]+rho[Lm]+rho[Lm+1])/3.;
    c2 = (rho[Hm-1]+rho[Hm]+rho[Hm+1])/3.;
    rhoL = (rholev[Lm-1]+rholev[Lm]+rholev[Lm+1])/3.;
    if(rhoL < 0.01) rhoL = 0.01;
    e1 = (a0+a1*(float)Lm)/1000.;
    e2 = (a0+a1*(float)Hm)/1000.;
    rhofg(Amass, e2, a, Tx, E1, E0x, isig, itemp, imodel, red, b1, b2);
    alpha = (log(rhoL)+log(c2)-log(eta*rhox)-log(c1))/(e1-e2);;
    Anorm = exp(-alpha*e1)*(rhoL)/c1;
    
    /* ******************************************* */
    /* Determine c1 and c2 by weighting all points */
    /* in each region with uncertainties and slope */
    /* ******************************************* */
    c1 = c1/anchorL();
    c2 = c2/anchorH();
    rhofg(Amass, e2, a, Tx, E1, E0x, isig, itemp, imodel, red, b1, b2 );
    alpha = (log(rhoL)+log(c2)-log(eta*rhox)-log(c1))/(e1-e2);
    Anorm = exp(-alpha*e1)*(rhoL)/c1;
    return 0;
}

float anchorL(){
    float corr, corrbest=0., sum, sumbest, sum0=0., cc, free, dc2, rhoL;
    corr = 0.25;
    sumbest = 1.0e+21;
    free = (float)L2 - (float)L1 - 1;
    if (free <= 0)free = 1.;
    for(j = 0; j <= 3750; j++){
        corr = corr + 0.001;
        sum  = 0.0;
        for(i = L1; i <= L2; i++){
            ex = (a0+a1*(float)i)/1000.;
            rhoL = rholev[i];
            if(rhoL < 0.01) rhoL = 0.01;
            cc = corr*Anorm*exp(alpha*ex)*rho[i];
            dc2= (corr*Anorm*exp(alpha*ex)*drho[i])*(corr*Anorm*exp(alpha*ex)*drho[i]);
            dc2 =sqrt(dc2*dc2 + 1.*1.);
            if(dc2 > 0) sum = sum + ((cc-rhoL)*(cc-rhoL)/dc2);
        }
        if(j == 499){
            sum0 = sum;
        }
        if(sum <= sumbest){
            sumbest = sum;
            corrbest= corr;
        }
    }
    chi2_lowx = sumbest;
    return corrbest;
}

float anchorH(){
    float corr, corrbest=0., sum, sumbest, sum0=0., cc, free, dc2;
    corr = 0.25;
    sumbest = 1.0e+21;
    free = (float)H2 - (float)H1 -1;
    if (free <= 0)free = 1.;
    for(j = 0; j <= 3750; j++){
        corr = corr + 0.001;
        sum  = 0.0;
        for(i = H1; i <= H2; i++){
            ex = (a0+a1*(float)i)/1000.;
            cc = corr*Anorm*exp(alpha*ex)*rho[i];
            dc2= (corr*Anorm*exp(alpha*ex)*drho[i])*(corr*Anorm*exp(alpha*ex)*drho[i]);
            dc2 =sqrt(dc2*dc2 + 1.*1.);
            rhofg(Amass, ex, a, Tx, E1, E0x, isig, itemp, imodel, red, b1, b2);
            if(dc2 > 0) sum=sum+(cc-eta*rhox)*(cc-eta*rhox)/dc2;
        }
        if(j == 499){
            sum0 = sum;
        }
        if(sum <= sumbest){
            sumbest = sum;
            corrbest= corr;
        }
    }
    chi2_highx = sumbest;
    return corrbest;
}

void nld_talys(){
    /* ******************************************* */
    /* Based on Cecilies ct_nld_talysform_92Zr.cpp */
    /* ******************************************* */
    double e_calc[56], temp_calc[56], nld_calc[56];
    /* ********************************************** */
    /* Double start_spin =  1/2 (half) or 0 (integer) */
    /* Loop up to 59/2 = 29.5 if half integer spin,   */
    /* up to 29 (starting from 0) if integer spin     */
    /* ********************************************** */
    int stop_spin  = 30;   // Spin loop stops after 30 iterations
    double n_cum   = 0.;   // Cumulative number of levels, different bin size on Ex!
    double ex_bin1 = 0.25; // 0.25 MeV from Ex=0.25 - 5.00 MeV, i= 0-19
    double ex_bin2 = 0.50; // 0.50 MeV from Ex=5.50 - 10.0 MeV, i=20-29
    double ex_bin3 = 1.00; // 1.00 MeV from Ex=11.0 - 20.0 MeV, i=30-39
    double ex_bin4 = 2.50; // 2.50 MeV from Ex=22.5 - 25.0 MeV, i=40-41
    double ex_bin5 = 5.00; // 5.00 MeV from Ex=25.0 - 30.0 MeV, i=41-42
    double ex_bin6 = 10.0; // 10.0 MeV from Ex=30.0 - 150. MeV, i=43-54
    double Ex;
    double I;
    double levdens;
    FILE *fp;
    
    /* ********************************** */
    /* Making the energy array for TALYS  */
    /* ********************************** */
    e_calc[0]=ex_bin1;
    for(i=0; i<55; i++){
        if(i>= 0 && i<19) e_calc[i+1] = e_calc[i] + ex_bin1;
        if(i>=19 && i<29) e_calc[i+1] = e_calc[i] + ex_bin2;
        if(i>=29 && i<39) e_calc[i+1] = e_calc[i] + ex_bin3;
        if(i>=39 && i<41) e_calc[i+1] = e_calc[i] + ex_bin4;
        if(i>=41 && i<42) e_calc[i+1] = e_calc[i] + ex_bin5;
        if(i>=42 && i<55) e_calc[i+1] = e_calc[i] + ex_bin6;
    }
    
    /* **************************************** */
    /* Calculate total NLD, cumulative NLD,     */
    /* and spin-dependent NLD and print to file */
    /* **************************************** */
    fp = fopen("talys_nld_cnt.txt","w");
    for(i=0; i<55; i++){
        nld_calc[i] = rhoexp(e_calc[i]);
        if(nld_calc[i]> 1.0e+30)break;  //rhotmopaw is float (and not double)
        if(i>= 0 && i<20) n_cum += nld_calc[i]*ex_bin1;
        if(i>=20 && i<30) n_cum += nld_calc[i]*ex_bin2;
        if(i>=30 && i<40) n_cum += nld_calc[i]*ex_bin3;
        if(i>=40 && i<42) n_cum += nld_calc[i]*ex_bin4;
        if(i>=42 && i<43) n_cum += nld_calc[i]*ex_bin5;
        if(i>=43 && i<55) n_cum += nld_calc[i]*ex_bin6;
        
        temp_calc[i]=temperature(Ex);

        fprintf(fp,"%7.2f %6.3f %9.2E %8.2E %8.2E ",e_calc[i],temp_calc[i],n_cum,nld_calc[i],nld_calc[i]);
        levdens = nld_calc[i];
        Ex  = e_calc[i];
        for(int j=0;j<stop_spin;j++){
            I = (double) j + start_spin;
            x = levdens*spin_distribution(Ex,I);
            fprintf(fp,"%8.2E ",x);
        }
        fprintf(fp,"\n");
    }
    fclose(fp);
}

void ReadSmooth(float resolution)
{
    float sigma = resolution/(2*sqrt(2*log(2)));

    // At this point we have already read all the discrete states.
    // Now we will set everything to zero and start over.
    for (i = 0 ; i < dim ; ++i){
        rholev[i] = 0;
    }

    for (i = 0 ; i < diml ; ++i){

        for (j = 0 ; j < dim ; ++j){
            emin = a0 + (j-0.5)*a1;
            emax = a0 + (j+0.5)*a1;

            double w0 = erf( (levenergy[i] - emax)/(sqrt(2)*sigma) );
            double w1 = erf( (levenergy[i] - emin)/(sqrt(2)*sigma) );
            //printf("%2.3f %2.3f\n", (levenergy[i] - emax)/(sqrt(2)*sigma), (levenergy[i] - emin)/(sqrt(2)*sigma));
            double weight = 0.5*(w1 - w0)/(a1*1e-3);

            

            rholev[j] += weight;
        }
    }
}

double spin_distribution(double Ex, double I){
    double g_Ex = 0., s2;
    int ii,i1,i2;
    for(ii=0; ii < FGmax;ii++){
        if((a0 + a1*ii)/1000. > Ex){
            break;
        }
    }
    i2 = ii;      //Interpolate to find sigma2 at Ex
    i1 = i2-1;
    s2 = sigma2[i1] + ((sigma2[i2]-sigma2[i1])/(a1/1000.))*(Ex-((a0 + a1*i1)/1000.));
//    printf("\n Ex = %6.3f  i1 = %d i2 = %d ex1= %6.3f ex2 =%6.3f, sigma2= %6.3f ",Ex,i1,i2,(a0 + a1*i1)/1000., (a0 + a1*i2)/1000.,s2);
    if(s2 < 1.) s2 = 1.;
    g_Ex = (2.*I+1.)*exp(-pow((I+0.5),2.)/(2.*s2))/(2.*s2);
    if(g_Ex>1.E-20) return g_Ex;
    else return 0.;
}

double rhoexp(double Ex){
    double rhox;
    int ii,i1,i2;
    for(ii=0; ii < FGmax;ii++){
        if((a0 + a1*ii)/1000. > Ex){
            break;
        }
    }
    i2 = ii;       //Interpolate to find rhoobs at Ex
    i1 = i2-1;
    rhox = rhoobs[i1] + ((rhoobs[i2]-rhoobs[i1])/(a1/1000.))*(Ex-((a0 + a1*i1)/1000.));
//    printf("\n Ex = %6.3f  i1 = %d i2 = %d ex1= %6.3f ex2 =%6.3f, rho= %6.3e ",Ex,i1,i2,(a0 + a1*i1)/1000., (a0 + a1*i2)/1000.,rhox);
    return rhox;
}

double temperature(double Ex){
    double temp=0.,temp0=0., tempL=0., tempH=0.;
    int ii,i1,i2,num=0;
    for(ii=0; ii < FGmax;ii++){
        if((a0 + a1*ii)/1000. > Ex){
            break;
        }
    }
    i2 = ii;       //Interpolate to find temperature at Ex, use 4 points total
    i1 = i2-1;
    if(rhoobs[i1] >0. && rhoobs[i2] > 0.){
        if(log(rhoobs[i2]) > log(rhoobs[i1])){
            temp0 = (a1/1000.)/(log(rhoobs[i2]) - log(rhoobs[i1]));
            num = num +1;
        }
    }
    if(i1 > 0){
        if(rhoobs[i1-1] >0. && rhoobs[i2-1] > 0.){
            if(log(rhoobs[i2-1]) > log(rhoobs[i1-1])){
                tempL = (a1/1000.)/(log(rhoobs[i2-1]) - log(rhoobs[i1-1]));
                num = num +1;
            }
        }
    }
    if(i2 < MAXDIM-1){
        if(rhoobs[i1+1] >0. && rhoobs[i2+1] > 0.){
            if(log(rhoobs[i2+1]) > log(rhoobs[i1+1])){
                tempH = (a1/1000.)/(log(rhoobs[i2+1]) - log(rhoobs[i1+1]));
                num = num +1;
            }
        }
    }
    if (num > 0)temp = (tempL + temp0 + tempH)/((float)num);
        //    printf("\n Ex = %6.3f  i1 = %d i2 = %d ex1= %6.3f ex2 =%6.3f, rho= %6.3e ",Ex,i1,i2,(a0 + a1*i1)/1000., (a0 + a1*i2)/1000.,rhox);
    return temp;
}













