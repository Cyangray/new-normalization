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
   TH2F *h = new TH2F("h"," ",10,-0.756000,7.820000,50,0.475174,138672928.000000);
   ifstream rholev("rholev.cnt"), rhopaw("rhopaw.cnt"), fermi("fermigas.cnt");
   float levels[64],rho[64],rhoerr[64],energy[783],energyerr[783],fermigas[783];
   float Bn[1]={7.280000};
   float Bnerr[1]={0.001};
   float rho_Bn[1]={13867293.000000};
   float rho_Bnerr[1]={12789092.000000};
   int i = 0;
   float a0 =  -0.7560;
   float a1 =   0.1280;
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
   	if(i<63){
   		rho[i]=y;
   	}
   	else{rhoerr[i-63]=y;}
   	i++;
   }
  	i=0;
	while(rholev){
		rholev >> z;
		levels[i]=z;
		i++;
  }
   TGraphErrors *rhoexp = new TGraphErrors(63,energy,rho,energyerr,rhoerr);
   TGraphErrors *rhoBn = new TGraphErrors(1,Bn,rho_Bn,Bnerr,rho_Bnerr);
   TGraph *fermicalc = new TGraph(782,energy,fermigas);
   TGraph *level = new TGraph(63,energy,levels);
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
   fermicalc->DrawGraph(38,&fermicalc->GetX()[30],&fermicalc->GetY()[30],"L");
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
   t.DrawLatex(    6.256,6.934e+07,"^{xx}Yy");
   TArrow *arrow1 = new TArrow(-0.116000,204.258668,-0.116000,31.331511,0.02,">");
   arrow1->Draw();
   TArrow *arrow2 = new TArrow(0.652000,1619.730244,0.652000,248.452591,0.02,">");
   arrow2->Draw();
   TArrow *arrow3 = new TArrow(3.084000,314168.854469,3.084000,48190.781250,0.02,">");
   arrow3->Draw();
   TArrow *arrow4 = new TArrow(5.260000,15863038.723841,5.260000,2433252.750000,0.02,">");
   arrow4->Draw();
   c1->Update();
   c1->Print("counting.pdf");
   c1->Print("counting.eps");
   c1->Print("counting.ps");
}
