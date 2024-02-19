{
   gROOT->Reset();
   gROOT->SetStyle("Plain");
   gStyle->SetOptTitle(0);
   gStyle->SetOptStat(0);
   gStyle->SetFillColor(0);
   gStyle->SetPadBorderMode(0);
   m = (TH1F*)gROOT->FindObject("h");
   if (m) m->Delete();
   TCanvas *c1 = new TCanvas("c1","Normalization of level density",600,600);
   TH2F *h = new TH2F("h"," ",10,-0.843336,6.727608,50,12.885845,24977216.000000);
   ifstream rholev("rholev.cnt"), rhopaw("rhopaw.cnt"), fermi("fermigas.cnt");
   float levels[59],rho[59],rhoerr[59],energy[820],energyerr[820],fermigas[820];
   float Bn[1]={6.243640};
   float Bnerr[1]={0.001};
   float rho_Bn[1]={2497721.500000};
   float rho_Bnerr[1]={172386.218750};
   int i = 0;
   float a0 =  -0.8433;
   float a1 =   0.1221;
   float x,y,z;
   while(fermi){
   	fermi >> x;
   	fermigas[i]=x;
   	energy[i]=a0+(a1*i);
   	energyerr[i]=0.0;
      i++;
   }
   i=0;
   while(rhopaw){
   	rhopaw >> y;
   	if(i<58){
   		rho[i]=y;
   	}
   	else{rhoerr[i-58]=y;}
   	i++;
   }
  	i=0;
	while(rholev){
		rholev >> z;
		levels[i]=z;
		i++;
  }
   TGraphErrors *rhoexp = new TGraphErrors(58,energy,rho,energyerr,rhoerr);
   TGraphErrors *rhoBn = new TGraphErrors(1,Bn,rho_Bn,Bnerr,rho_Bnerr);
   TGraph *fermicalc = new TGraph(819,energy,fermigas);
   TGraph *level = new TGraph(58,energy,levels);
   c1->SetLogy();
   c1->SetLeftMargin(0.14);
   h->GetXaxis()->CenterTitle();
   h->GetXaxis()->SetTitle("Excitation energy E (MeV)");
   h->GetYaxis()->CenterTitle();
   h->GetYaxis()->SetTitleOffset(1.4);
   h->GetYaxis()->SetTitle("Level density #rho (E) (MeV^{-1})");
   h->Draw();
   rhoexp->SetMarkerStyle(21);   rhoexp->SetMarkerSize(0.8);
   rhoexp->Draw("P");
   fermicalc->SetLineStyle(2);
   fermicalc->DrawGraph(32,&fermicalc->GetX()[31],&fermicalc->GetY()[31],"L");
   level->SetLineStyle(1);
   level->Draw("L");
   rhoBn->SetMarkerStyle(25);
   rhoBn->SetMarkerSize(0.8);
   rhoBn->Draw("P");
   TLegend *leg = new TLegend(0.15,0.70,0.6,0.85);
   leg->SetBorderSize(0);
   leg->SetFillColor(0);
   leg->AddEntry(rhoexp," Oslo data ","P");
   leg->AddEntry(level," Known levels ","L");
   leg->AddEntry(fermicalc," CT or FG model ","L");	
   leg->AddEntry(rhoBn," #rho from neutron res. data ","P");
   leg->Draw();
   TLatex t;
   t.SetTextSize(0.05);
   t.DrawLatex(    5.382,1.249e+07,"^{xx}Yy");
   TArrow *arrow1 = new TArrow(0.499896,3322.531480,0.499896,604.943359,0.02,">");
   arrow1->Draw();
   TArrow *arrow2 = new TArrow(0.744120,5725.857338,0.744120,1042.524170,0.02,">");
   arrow2->Draw();
   TArrow *arrow3 = new TArrow(3.064248,463958.158124,3.064248,84474.265625,0.02,">");
   arrow3->Draw();
   TArrow *arrow4 = new TArrow(4.651704,5979674.205491,4.651704,1088737.375000,0.02,">");
   arrow4->Draw();
   c1->Update();
   c1->Print("counting.pdf");
   c1->Print("counting.eps");
   c1->Print("counting.ps");
}
