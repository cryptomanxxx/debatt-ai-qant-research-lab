"""Compile a reviewed AI Researcher proposal into an approved guarded job."""
import json, re, sys
from pathlib import Path

if len(sys.argv)!=2:
    raise SystemExit("Usage: python scripts/approve_proposal.py <proposal_id>")

proposal_id=sys.argv[1]
if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,95}",proposal_id):
    raise SystemExit("Invalid proposal id")

p=(Path("pnn-v1/proposals")/f"{proposal_id.replace('pnn-v1-','')}.json") if proposal_id.startswith("pnn-v1-") else (Path("research_queue/proposals")/f"{proposal_id}.json")
if not p.exists(): raise SystemExit(f"Unknown proposal: {proposal_id}")
proposal=json.loads(p.read_text())
if proposal.get("proposal_id")!=proposal_id or proposal.get("status")!="proposed":
    raise SystemExit("Proposal is not eligible for approval")
if proposal.get("approval_required") is not True:
    raise SystemExit("Proposal does not declare human approval")

# Explicit allowlist: proposals cannot inject executable paths or commands.
MAPPINGS={
 "pnn-v1-proposal001":(
   "pnn-v1-exp001","pnn-v1/experiments/exp001/run.py"),
 "pnn-v1-proposal002":(
   "pnn-v1-exp002","pnn-v1/experiments/exp002/run.py"),
 "pnn-v1-proposal003":(
   "pnn-v1-exp003","pnn-v1/experiments/exp003/run.py"),
 "pnn-v1-proposal004":(
   "pnn-v1-exp004","pnn-v1/experiments/exp004/run.py"),
 "pnn-v1-proposal005":(
   "pnn-v1-exp005","pnn-v1/experiments/exp005/run.py"),
 "pnn-v1-proposal006":(
   "pnn-v1-exp006","pnn-v1/experiments/exp006/run.py"),
 "pnn-v1-proposal007":(
   "pnn-v1-exp007","pnn-v1/experiments/exp007/run.py"),
 "pnn-v1-proposal008":(
   "pnn-v1-exp008","pnn-v1/experiments/exp008/run.py"),
 "pnn-v1-proposal009":(
   "pnn-v1-exp009","pnn-v1/experiments/exp009/run.py"),
 "pnn-v1-proposal010":(
   "pnn-v1-exp010","pnn-v1/experiments/exp010/run.py"),
 "pnn-v1-proposal011":(
   "pnn-v1-exp011","pnn-v1/experiments/exp011/run.py"),
 "pnn-v1-proposal012":(
   "pnn-v1-exp012","pnn-v1/experiments/exp012/run.py"),
 "pnn-v1-proposal013":(
   "pnn-v1-exp013","pnn-v1/experiments/exp013/run.py"),
 "pnn-v1-proposal014":(
   "pnn-v1-exp014","pnn-v1/experiments/exp014/run.py"),
 "pnn-v1-proposal015":(
   "pnn-v1-exp015","pnn-v1/experiments/exp015/run.py"),
 "pnn-v1-proposal016":(
   "pnn-v1-exp016","pnn-v1/experiments/exp016/run.py"),
 "pnn-v1-proposal017":(
   "pnn-v1-exp017","pnn-v1/experiments/exp017/run.py"),
 "pnn-v1-proposal018":(
   "pnn-v1-exp018","pnn-v1/experiments/exp018/run.py"),
 "pnn-v1-proposal019":(
   "pnn-v1-exp019","pnn-v1/experiments/exp019/run.py"),
 "pnn-v1-proposal003":(
   "pnn-v1-exp003","pnn-v1/experiments/exp003/run.py"),
 "proposal-001-replicate-expanding-topologies":(
   "exp006-multiseed-replication","experiments/006_multiseed_replication/run.py"),
 "proposal-002-expansion-ratio":(
   "exp007-expansion-ratio","experiments/007_expansion_ratio/run.py"),
 "proposal-003-stronger-mnist-validation":(
   "exp008-full-mnist-validation","experiments/008_full_mnist_validation/run.py"),
 "proposal-004-cross-dataset-validation":(
   "exp009-cross-dataset-validation","experiments/009_cross_dataset_validation/run.py"),
 "proposal-005-qant-fourier-layer":(
   "exp010-qant-fourier-layer","experiments/010_qant_fourier_layer/run.py"),
 "proposal-006-qant-fourier-capacity":(
   "exp011-qant-fourier-capacity","experiments/011_qant_fourier_capacity/run.py"),
 "proposal-007-qant-fourier-robust-training":(
   "exp012-qant-fourier-robust-training","experiments/012_qant_fourier_robust_training/run.py"),
 "proposal-008-qant-fourier-mismatch-diagnostic":(
   "exp013-qant-fourier-mismatch-diagnostic","experiments/013_qant_fourier_mismatch_diagnostic/run.py"),
 "proposal-009-qant-fourier-error-decomposition":(
   "exp014-qant-fourier-error-decomposition","experiments/014_qant_fourier_error_decomposition/run.py"),
 "proposal-010-qant-aligned-training-surrogate":(
   "exp015-qant-aligned-training-surrogate","experiments/015_qant_aligned_training_surrogate/run.py"),
 "proposal-011-qant-activation-error-occupancy":(
   "exp016-qant-activation-error-occupancy","experiments/016_qant_activation_error_occupancy/run.py"),
 "proposal-012-qant-logit-margin-diagnostic":(
   "exp017-qant-logit-margin-diagnostic","experiments/017_qant_logit_margin_diagnostic/run.py"),
 "proposal-013-qant-frequency-controlled-g8":(
   "exp018-qant-frequency-controlled-g8","experiments/018_qant_frequency_controlled_g8/run.py"),
 "proposal-014-qant-amplitude-error-accumulation":(
   "exp019-qant-amplitude-error-accumulation","experiments/019_qant_amplitude_error_accumulation/run.py"),
 "proposal-015-qant-kan-additivity-ladder":(
   "exp020-qant-kan-additivity-ladder","experiments/020_qant_kan_additivity_ladder/run.py"),
 "proposal-016-qant-accumulation-scaling-law":(
   "exp021-qant-accumulation-scaling-law","experiments/021_qant_accumulation_scaling_law/run.py"),
 "proposal-017-qant-grouped-accumulation-architecture":(
   "exp022-qant-grouped-accumulation-architecture","experiments/022_qant_grouped_accumulation_architecture/run.py"),
 "proposal-018-qant-hierarchical-low-fanin":(
   "exp023-qant-hierarchical-low-fanin","experiments/023_qant_hierarchical_low_fanin/run.py"),
 "proposal-019-qant-hierarchical-architecture-search":(
   "exp024-qant-hierarchical-architecture-search","experiments/024_qant_hierarchical_architecture_search/run.py"),
 "proposal-020-qant-hierarchical-error-localization":(
   "exp025-qant-hierarchical-error-localization","experiments/025_qant_hierarchical_error_localization/run.py"),
 "proposal-021-qant-predictive-compatibility-model":(
   "exp026-qant-predictive-compatibility-model","experiments/026_qant_predictive_compatibility_model/run.py"),
 "proposal-022-qant-factorial-accumulation-map":(
   "exp027-qant-factorial-accumulation-map","experiments/027_qant_factorial_accumulation_map/run.py"),
 "proposal-023-qant-predictive-compatibility-v2":(
   "exp028-qant-predictive-compatibility-v2","experiments/028_qant_predictive_compatibility_v2/run.py"),
 "proposal-024-qant-7x2-residual-localization":(
   "exp029-qant-7x2-residual-localization","experiments/029_qant_7x2_residual_localization/run.py"),
 "proposal-025-qant-compatibility-model-v3":(
   "exp030-qant-compatibility-model-v3","experiments/030_qant_compatibility_model_v3/run.py")
}
if proposal_id not in MAPPINGS:
    raise SystemExit("No reviewed compiler mapping exists for this proposal")

job_id,experiment=MAPPINGS[proposal_id]
job={
 "job_id":job_id,
 "status":"approved",
 "experiment":experiment,
 "runner":"github-actions",
 "description":"Human-approved compilation of "+proposal_id,
 "budget":proposal["requested_budget"],
 "source_proposal":proposal_id
}
out=Path("research_queue/jobs")/(job_id+".json")
out.write_text(json.dumps(job,indent=2)+"\n")
print(json.dumps(job,indent=2))
