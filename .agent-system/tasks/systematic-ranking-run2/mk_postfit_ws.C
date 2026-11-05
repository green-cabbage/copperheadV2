// Transplant the VBF sideband post-fit nuisance values into the combined
// VBF+ggH workspace and save them as a snapshot, replicating
// scripts/produce_significance.sh L76-86. assignValueOnly matches by NAME, so
// ggH-only nuisances and the SR prop_bin* parameters keep their pre-fit values
// because the sideband model has no parameter of that name.
void mk_postfit_ws(const char* sbfit, const char* comb, const char* out,
                   const char* snap, const char* dump) {
  TFile fsb(sbfit);
  RooWorkspace* wsb = (RooWorkspace*)fsb.Get("w");
  if (!wsb) { printf("FATAL no workspace in sb fit\n"); return; }
  wsb->loadSnapshot("MultiDimFit");
  RooStats::ModelConfig* mcsb = (RooStats::ModelConfig*)wsb->genobj("ModelConfig");
  const RooArgSet* nus = mcsb->GetNuisanceParameters();

  TFile fc(comb);
  RooWorkspace* wc = (RooWorkspace*)fc.Get("w");
  if (!wc) { printf("FATAL no workspace in combined file\n"); return; }
  RooArgSet full(wc->allVars());

  // keep the pre-fit point so the written file's CURRENT values stay pre-fit and
  // only the snapshot carries the sideband values
  wc->saveSnapshot("PreFit", full);

  int matched = 0, unmatched = 0;
  FILE* fp = fopen(dump, "w");
  for (auto a : *nus) {
    RooRealVar* v = dynamic_cast<RooRealVar*>(a); if (!v) continue;
    RooAbsArg* t = full.find(v->GetName());
    if (t) { matched++; fprintf(fp, "%s %.10g\n", v->GetName(), v->getVal()); }
    else   { unmatched++; printf("SB-ONLY %s\n", v->GetName()); }
  }
  fclose(fp);

  full.assignValueOnly(*nus);
  wc->saveSnapshot(snap, full);
  wc->loadSnapshot("PreFit");   // restore current values to pre-fit before writing
  wc->writeToFile(out);
  printf("matched=%d sb_only=%d comb_allVars=%d snapshot=%s out=%s\n",
         matched, unmatched, full.getSize(), snap, out);
}
