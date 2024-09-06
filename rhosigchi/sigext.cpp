{
   gROOT->Reset();
   gROOT->SetStyle("Plain");
   gStyle->SetOptTitle(0);
   gStyle->SetOptStat(0);
   gStyle->SetFillColor(0);
   gStyle->SetPadBorderMode(0);
   m = (TH1F*)gROOT->FindObject("h");
   if (m) m->Delete();
   TCanvas *c1 = new TCanvas("c1","Normalization of gamma-transmission coefficient",600,600);
   TH2F *h = new TH2F("h"," ",10,-0.843336,   7.117,50,7.393e+01,9.493e+07);
   ifstream sigfile("sigpaw.cnt");
   float sig[59],sigerr[59];
   float energy[164],energyerr[164];
   float extL[165],extH[165];
   int i;
   float a0 = -0.8433;
   float a1 =  0.1221;
   for(i = 0; i < 63; i++){
   	energy[i] = a0 + (a1*i);
   	energyerr[i] = 0.0;
   	extL[i] = 0.0;
   	extH[i] = 0.0;
   }
   float x, y;
   i = 0;
   while(sigfile){
   	sigfile >> x;
   	if(i<58){
   		sig[i]=x;
   	}
   	else{sigerr[i-58]=x;}
   	i++;
   }
   ifstream extendfile("extendLH.cnt");
   i = 0;
   while(extendfile){
   	extendfile >> x >> y ;
   	extL[i]=x;
   	extH[i]=y;
   	i++;
   }
   TGraph *extLgraph = new TGraph(63,energy,extL);
   TGraph *extHgraph = new TGraph(63,energy,extH);
   TGraphErrors *sigexp = new TGraphErrors(58,energy,sig,energyerr,sigerr);
   c1->SetLogy();
   c1->SetLeftMargin(0.14);
   h->GetXaxis()->CenterTitle();
   h->GetXaxis()->SetTitle("#gamma-ray energy E_{#gamma} (MeV)");
   h->GetYaxis()->CenterTitle();
   h->GetYaxis()->SetTitleOffset(1.4);
   h->GetYaxis()->SetTitle("Transmission coeff. (arb. units)");
   h->Draw();
   sigexp->SetMarkerStyle(21);
   sigexp->SetMarkerSize(0.8);
   sigexp->Draw("P");
   extLgraph->SetLineStyle(1);
   extLgraph->DrawGraph(20,&extLgraph->GetX()[0],&extLgraph->GetY()[0],"L");
   extHgraph->SetLineStyle(1);
   extHgraph->DrawGraph(22,&extHgraph->GetX()[41],&extHgraph->GetY()[41],"L");
   TArrow *arrow1 = new TArrow(1.355e+00,2.913e+04,1.355e+00,4.640e+03,0.02,">");
   arrow1->Draw();
   TArrow *arrow2 = new TArrow(1.477e+00,5.034e+04,1.477e+00,8.020e+03,0.02,">");
   arrow2->Draw();
   TArrow *arrow3 = new TArrow(4.163e+00,4.241e+06,4.163e+00,6.757e+05,0.02,">");
   arrow3->Draw();
   TArrow *arrow4 = new TArrow(4.652e+00,8.445e+06,4.652e+00,1.345e+06,0.02,">");
   arrow4->Draw();
   c1->Update();
   c1->Print("sigext.pdf");
   c1->Print("sigext.eps");
   c1->Print("sigext.ps");
}
