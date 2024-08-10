{
	gROOT->Reset();
	gROOT->SetStyle("Plain");
   gStyle->SetOptTitle(0);
   gStyle->SetOptStat(0);
   gStyle->SetFillColor(0);
   gStyle->SetPadBorderMode(0);
   m = (TH1F*)gROOT->FindObject("h");
   if (m) m->Delete();
   TCanvas *c1 = new TCanvas("c1","Spincut parameter",600,600);
   TH2F *h = new TH2F("h"," ",10,-0.756000,23.052000,50,0.001,    7.9);
   ifstream spincutfile("spincut.cnt");
   float energy[188],spincut[188];
   int i = 0;
   float a0 =  -0.7560;
   float a1 =   0.1280;
   float x,y,z;
   while(spincutfile){
   	spincutfile >> x;
	   spincut[i]=x;
	   energy[i]=a0+(a1*i);
	   i++;
   }
   TGraph *spincutgraph = new TGraph(187,energy,spincut);
   c1->SetLeftMargin(0.14);
   h->GetXaxis()->CenterTitle();
   h->GetXaxis()->SetTitle("Excitation energy E (MeV)");
   h->GetYaxis()->CenterTitle();
   h->GetYaxis()->SetTitleOffset(1.4);
   h->GetYaxis()->SetTitle("Spin cutoff #sigma");
   h->Draw();
   spincutgraph->Draw("L");
   c1->Update();
   c1->Print("spincut.pdf");
   c1->Print("spincut.eps");
   c1->Print("spincut.ps");
}
