"""CPU-only counterfactual admission; never imported by the GPU worker.

Prior art: GRM A-DEC/RT1 (contributors, 2026), reuse real predicates, plan,
margin constant and stable split demotion. Boolean lexical retrieval: Manning,
Raghavan, Schuetze (2008), IIR; unverified — lead to check title/author.
New: lead's Rule1/Rule2 compositions; no prior art known to me for exact mix.
"""
import copy
import math
import re
from core import grm_admission as adm
from core.graft_arena import ArenaCache


def shaped_tokens(question):
    # Prior art: core _rare_tokens and _query_content_tokens (GRM, 2026).
    # Borrow normalization and stop list. Add title-case/hyphen shape per lead;
    # this is a lexical proxy for names, NOT a named-entity recognizer.
    normalized = ArenaCache._norm_text(question)
    selected = set(ArenaCache._rare_tokens(question))
    for raw in re.findall(r'[A-Za-z0-9][\w:.,\-]*', normalized):
        raw = raw.rstrip('.,:;'); low = raw.casefold()
        if low in ArenaCache._QUERY_LEX_STOP or len(low) < 2:
            continue
        if raw[:1].isupper() or '-' in raw:
            selected.add(low)
    ordered = list(dict.fromkeys(w for w in adm.normalized_words(normalized) if w in selected))
    return ordered


def exact_scores(arena, question, eligible, ranking):
    # Prior art: decisive_admission_profile exact reconstruction (GRM, 2026).
    # Same score law, forced evaluation for Rule2 even when Rule0 skips it.
    key = arena._probe_key(question)
    base = arena._vector_route_scores(key, eligible)
    if base is None:
        base = {i:float(arena._cent_score(key, arena.grafts[i])) for i in eligible}
        base = {i:s for i,s in base.items() if math.isfinite(s)}
    base = arena._normalize_scores(arena._length_debias_scores(base, eligible)) or {}
    qlex = arena._query_lex_tokens(question)
    scores = {i:float(base[i])+float(arena._lex_bonus(qlex, arena.grafts[i]))
              for i in eligible if i in base and math.isfinite(float(base[i]))}
    reference = sorted(scores, key=lambda i:(-scores[i],i))
    if reference[:len(ranking)] != ranking:
        raise ValueError('OFFLINE_SCORE_RECONSTRUCTION_MISMATCH')
    return scores


def snapshot(arena, question, exclude):
    profile = adm.decisive_admission_profile(arena, question, exclude=exclude, route_limit=6)
    eligible = [i for i in arena._route_cand_base() if i not in set(exclude)]
    ranking = profile['admission_ranking_before_demotion']
    scores = exact_scores(arena, question, eligible, ranking)
    nodes = [{k:copy.deepcopy(g.get(k)) for k in ('text','kind','retired','ntok','sources','metadata')}
             for g in arena.grafts]
    return dict(question=question,nodes=nodes,eligible=eligible,ranking=ranking,
        scores={str(k):v for k,v in scores.items()},split_members=profile['admission_split_family_ids'],
        margin=(scores[ranking[0]]-scores[ranking[1]]) if len(ranking)>1 else 0.0,
        exclude=list(exclude),baseline_profile=profile,evidence_class='CPU fake routing; real core admission; frozen pre-probe state')


def evaluate(s, rule):
    """Admission plan only: no fit, mount, reader, or mutation of frozen state."""
    q=s['question']; nodes=s['nodes']; eligible=s['eligible']; ranking=list(s['ranking'])
    ordered,rare=adm.ordered_identifier_tokens(ArenaCache, q)
    if rule == 1:
        ordered=shaped_tokens(q); rare=set(ordered)
    if rule in (0,1):
        hits=[i for i in eligible if adm.is_identifier_binding(candidate_text=nodes[i]['text'] or '',
                  ordered_identifier_tokens=ordered, rare_identifier_tokens=rare)]
        ranking,demoted=adm.demote_non_binding_split_members(ranking=ranking,binding=hits,
                  split_members=s.get('split_members',[]))
        hits=[i for i in ranking if i in hits]+sorted(set(hits)-set(ranking))
        # Some real receipts omit an irrelevant margin. It is only required
        # by A-DEC for one off-rank binder; do not substitute zero there.
        margin=s.get('margin')
        if margin is None and len(hits)==1 and ranking and hits[0]!=ranking[0]:
            return dict(status='UNRESOLVED',reason='MISSING_DECISION_RELEVANT_MARGIN')
        plan,branch=adm.policy_plan(ranking=ranking,identified_candidates=hits,route_margin_1_2=margin or 0.0)
        profile=dict(identifier_tokens=ordered,identified_candidates=hits)
        refusal=adm.identifier_unbound_abstention(profile)
        return dict(status='EVALUATED',rule=rule,identifier_tokens=ordered,identified_candidates=hits,
                    ranking=ranking,plan=[] if refusal else plan,proposed_plan=plan,
                    mounts=bool(plan and not refusal),refused=bool(refusal),branch=branch,demoted=demoted)
    if rule != 2: raise ValueError('UNKNOWN_RULE')
    margin=s.get('margin')
    if not ranking:
        return dict(status='EVALUATED',rule=2,plan=[],mounts=False,refused=False,branch='empty_ranking')
    if margin is None:
        return dict(status='UNRESOLVED',reason='MISSING_EXACT_MARGIN')
    if margin > adm.MARGIN_THRESHOLD:
        plan=ranking[:1]; branch='fit_margin_decisive_rank1'
    else:
        # Prior art: stable tie breaking, core RT1 stable partitions (2026).
        # Lead's identifiers break EXACT rank1 score ties only, never a gate.
        ordered=shaped_tokens(q)
        hits={i for i in eligible if ordered and adm.is_identifier_binding(candidate_text=nodes[i]['text'] or '',
                  ordered_identifier_tokens=ordered,rare_identifier_tokens=ordered)}
        scores=s.get('scores')
        if margin == 0.0 and len(ranking)>1 and not scores and hits.intersection(ranking):
            return dict(status='UNRESOLVED',reason='MISSING_TIED_SCORE_GROUP')
        if scores:
            tied=[i for i in ranking if scores.get(str(i))==scores.get(str(ranking[0]))]
            ranking=[i for i in tied if i in hits]+[i for i in tied if i not in hits]+[i for i in ranking if i not in tied]
        plan=ranking[:3];branch='margin_insurance_k3_identifier_tiebreak'
    return dict(status='EVALUATED',rule=2,plan=plan,mounts=bool(plan),refused=False,branch=branch)
