void Branchdump(){
	TFile *file = TFile::Open("/isilon/hadoop/store/user/dryu/DijetSkim/v2_0_7/SingleMuon2016/SingleMuon/SingleMuon_Run2016B2/201215_061531/0000/nanoskim_1.root");
	TTree *tree = (TTree*)file->Get("Events");
	tree->Print();
}