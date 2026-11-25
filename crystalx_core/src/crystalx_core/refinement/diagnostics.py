"""Answer-free summaries of a completed immutable refinement model."""
from collections import Counter,defaultdict
import math
from statistics import median
from crystalx_core.refinement.adaptive import AdaptivePolicy,geometry_summary
from crystalx_core.refinement.cycle import displacement_value


def distribution(values):
    values=sorted(v for v in values if v is not None and math.isfinite(v))
    return {"count":len(values),"minimum":values[0] if values else None,
        "median":median(values) if values else None,"p90":values[min(len(values)-1,int(.9*(len(values)-1)))] if values else None,
        "maximum":values[-1] if values else None}


def model_diagnostics(model,*,policy:AdaptivePolicy,include_geometry=True):
    values=[displacement_value(model,a) for a in model.atoms];groups=defaultdict(list)
    for atom in model.atoms:groups[model.element_of(atom)].append(displacement_value(model,atom))
    result={"model_sha256":model.sha256,"n_sites":len(model.atoms),
        "element_counts":dict(sorted(Counter(model.element_of(a) for a in model.atoms).items())),
        "r1_observed":model.r_factors.r1_observed,"r1_all":model.r_factors.r1_all,
        "wR2":model.r_factors.wr2,"goof":model.r_factors.goof,
        "n_parameters":model.r_factors.n_parameters,"n_observed":model.r_factors.n_observed,
        "residual_peak_maximum":model.r_factors.highest_peak,"residual_peak_minimum":model.r_factors.deepest_hole,
        "u_iso_distribution":distribution(values),"u_by_element":{e:distribution(v) for e,v in sorted(groups.items())},
        "displacement_scalar":"Uiso_or_cell_metric_Ueq; raw tensors retained",
        "anisotropic_site_count":sum(a.is_anisotropic for a in model.atoms),
        "nonpositive_u_count":sum(v is not None and v<=0 for v in values),
        "high_u_count":sum(v is not None and v>=policy.u_abs_high for v in values),
        "high_u_threshold":policy.u_abs_high,"expert_used":False,
        "scientifically_validated":False}
    if include_geometry:result["geometry"]=geometry_summary(model,policy=policy)
    return result
