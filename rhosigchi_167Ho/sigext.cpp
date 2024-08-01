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
   TH2F *h = new TH2F("h"," ",10,-0.756000,   8.180,50,9.799e+00,3.969e+07);
   ifstream sigfile("sigpaw.cnt");
   float sig[64],sigerr[64];
   float energy[157],energyerr[157];
   float extL[158],extH[158];
   int i;
   float a0 = -0.7560;
   float a1 =  0.1280;
   for(i = 0; i < 68; i++){
   	energy[i] = a0 + (a1*i);
   	energyerr[i] = 0.0;
   	extL[i] = 0.0;
   	extH[i] = 0.0;
   }
   float x, y;
   i = 0;
   while(sigfile){
   	sigfile >> x;
   	if(i<63){
   		sig[i]=x;
   	}
   	else{sigerr[i-63]=x;}
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
   TGraph *extLgraph = new TGraph(68,energy,extL);
   TGraph *extHgraph = new TGraph(68,energy,extH);
   TGraphErrors *sigexp = new TGraphErrors(63,energy,sig,energyerr,sigerr);
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
   extLgraph->DrawGraph(25,&extLgraph->GetX()[0],&extLgraph->GetY()[0],"L");
   extHgraph->SetLineStyle(1);
   extHgraph->DrawGraph(18,&extHgraph->GetX()[50],&extHgraph->GetY()[50],"L");
   TArrow *arrow1 = new TArrow(1.292e+00,3.424e+03,1.292e+00,6.449e+02,0.02,">");
   arrow1->Draw();
   TArrow *arrow2 = new TArrow(2.316e+00,2.713e+04,2.316e+00,5.110e+03,0.02,">");
   arrow2->Draw();
   TArrow *arrow3 = new TArrow(5.644e+00,3.671e+06,5.644e+00,6.914e+05,0.02,">");
   arrow3->Draw();
   TArrow *arrow4 = new TArrow(6.924e+00,1.580e+07,6.924e+00,2.975e+06,0.02,">");
   arrow4->Draw();
   c1->Update();
   c1->Print("sigext.pdf");
   c1->Print("sigext.eps");
   c1->Print("sigext.ps");
}
