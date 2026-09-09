# RD2 source evidence

Line-numbered excerpts of unchanged local files; SOURCE_INTEGRITY.json verifies the RD1 source pins.

## scripts/grm_c7_register.py:25-45
```text
25:     def fact(t, entity, value, text=None, **kw):
26:         entities.add(entity)
27:         phrase = f'current {entity} value is {value}'
28:         events[t] = {'turn': t, 'kind': 'fact', 'fact_id': entity, 'value': value,
29:                      'fact_phrase': phrase, 'user': text or f'The {phrase}.',
30:                      'assistant': 'Recorded.', **kw}
31: 
32:     # Supporting sources precede every anchor. Correction history is verbatim
33:     # in the oracle, including superseded values; expected answers never enter
34:     # the memory query. Fresh fixture namespace is unrelated to C2 needles.
35:     fact(1, 'C7-Vesper-0', 'Mica-431',
36:          'The current C7-Vesper-0 value is Mica-431. The current C7-Vesper-1 value is Mica-432.')
37:     entities.add('C7-Vesper-1')
38:     fact(2, 'C7-Vesper-0', 'Flint-511',
39:          'Correction of both original records: the current C7-Vesper-0 value is Flint-511; '
40:          'the current C7-Vesper-1 value is Flint-512. The previous Mica values are obsolete.')
41:     fact(3, 'C7-Archive-0', 'Reed-611', 'For C7-Archive-0 the outbound tag is Reed-611.')
42:     fact(4, 'C7-Archive-1', 'Reed-612', 'For C7-Archive-1 the outbound tag is Reed-612.')
43:     fact(5, 'C7-AliasBase-0', 'Jasper-711',
44:          'The current C7-AliasBase-0 value is Jasper-711. The current C7-AliasBase-1 value is Jasper-712.')
45:     entities.add('C7-AliasBase-1')
```

## scripts/grm_c7_register.py:62-88
```text
62:         elif cls == 'alias':
63:             entity = f'C7-AliasBase-{j}'
64:             value = f'Jasper-{711+j}'
65:             alias = f'C7-Signal-{j}'
66:             fact(anchor, entity, value if answerable else 'UNSET',
67:                  f'{alias} is an alias for {entity}.')
68:             aliases.append({'turn': anchor, 'alias': alias, 'entity': entity})
69:             source = [5, anchor]
70:             question = f'What is the current value for {alias}?'
71:         elif cls == 'correction':
72:             entity = f'C7-Vesper-{j}'
73:             value = f'Onyx-{911+j}'
74:             fact(anchor, entity, value)
75:             events[anchor].update(kind='supersede', old_value=f'Flint-{511+j}',
76:                 correction_command=f'correct memory: current {entity} value is Flint-{511+j} => '
77:                                    f'The current {entity} value is {value}.')
78:             corrections.append({'turn': anchor, 'entity': entity, 'corrects_turn': 2,
79:                                 'earlier_correction': True})
80:             if not answerable:
81:                 # Negative control introduces an entity, not a fictitious
82:                 # correction of an unregistered predecessor.
83:                 events[anchor]['kind'] = 'fact'
84:                 events[anchor].pop('correction_command')
85:                 events[anchor].pop('old_value')
86:                 corrections.pop()
87:             source = [1, 2, anchor]
88:             question = f'After all corrections, what is the current {entity} value?'
```

## scripts/grm_e2e_session.py:85-95
```text
85: # default ArenaCache sink "<conversation>\n" mis-seats GPT-OSS YARN.
86: HARMONY_SINK = (
87:     "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
88:     "Valid channel: final.<|end|>"
89: )
90: SYSTEM_PREFIX = (
91:     "<|start|>system<|message|>You are ChatGPT. Reasoning: low. "
92:     "Valid channel: final.<|end|><|start|>user<|message|>"
93: )
94: ASSISTANT_FINAL = "<|end|><|start|>assistant<|channel|>final<|message|>"
95: HARMONY_STOPS = (
```

## scripts/grm_e2e_session.py:143-158
```text
143: def supersede_turn(fact_id: str, old: str, new: str) -> dict[str, Any]:
144:     old_phrase = f"current {fact_id} value is {old}"
145:     new_phrase = f"current {fact_id} value is {new}"
146:     event = {
147:         "kind": "supersede",
148:         "fact_id": fact_id,
149:         "old_value": old,
150:         "value": new,
151:         "user": (
152:             f"Authoritative update turn. fact {fact_id}. The {new_phrase}. "
153:             f"This is the current value. If asked later, answer {new} only."
154:         ),
155:         "correction_command": f"correct memory: {old_phrase} => The {new_phrase}.",
156:         "fact_phrase": new_phrase,
157:     }
158:     event["assistant"] = plant_acceptance(event)
```

## scripts/grm_e2e_session.py:172-192
```text
172: def probe_turn(
173:     fact_id: str,
174:     expected: str,
175:     *,
176:     source_turn: int,
177:     supersession: bool = False,
178:     old_value: str | None = None,
179: ) -> dict[str, Any]:
180:     return {
181:         "kind": "probe",
182:         "fact_id": fact_id,
183:         "expected": expected,
184:         "accepts": [expected],
185:         "old_value": old_value,
186:         "source_turn": int(source_turn),
187:         "supersession": bool(supersession),
188:         "user": (
189:             f"Recall probe. What is the current {fact_id} value? "
190:             "Reply with only the value."
191:         ),
192:     }
```

## scripts/grm_e2e_session.py:2655-2670
```text
2655:         if args.turn_pipeline == "single":
2656:             # Registered legacy path. Fact/supersede plants use feed()
2657:             # COMPLETE; probe/filler keep production chat()→step().
2658:             if event["kind"] in ("fact", "supersede"):
2659:                 answer, info, chat_node_id = plant_complete_turn(repo, event)
2660:                 plant_mode = info.get("plant_mode")
2661:                 if event["kind"] == "supersede":
2662:                     t0 = time.perf_counter()
2663:                     try:
2664:                         correction_result = repo.apply_memory_command(
2665:                             event["correction_command"])
2666:                     finally:
2667:                         correction_wall_ms = (
2668:                             time.perf_counter() - t0) * 1000.0
2669:                         timers.supersession_ms += correction_wall_ms
2670:                         timers.supersession_calls += 1
```

## core/graft_repository.py:1953-2001
```text
1953:     def correct_memory(self, query, replacement, **metadata):
1954:         supersedes = []
1955:         q = str(query or "").strip().lower()
1956:         before = self._snapshot_state()
1957:         if not q:
1958:             # Same law as forget(): an empty/whitespace query never means
1959:             # "every node". The correction is written with no supersedes.
1960:             targets = []
1961:         else:
1962:             targets = self._native_active_text_matches(q)
1963:         if targets is None:
1964:             targets = []
1965:             for i, g in enumerate(self.arena.grafts):
1966:                 if q in g.get("text", "").lower():
1967:                     targets.append(i)
1968:         active_targets = []
1969:         for i in targets:
1970:             g = self.arena.grafts[int(i)]
1971:             meta = g.setdefault("metadata", self._default_metadata(g))
1972:             if meta.get("active", True):
1973:                 active_targets.append(int(i))
1974:         mutation_plan = self._native_memory_mutation_plan(
1975:             "correct", has_query=bool(q), target_count=len(active_targets),
1976:             has_replacement=replacement is not None)
1977:         if (mutation_plan is None
1978:                 or mutation_plan.get("action") == "supersede_targets"):
1979:             supersedes = list(active_targets)
1980:         else:
1981:             supersedes = []
1982:         for i in supersedes:
1983:             g = self.arena.grafts[int(i)]
1984:             meta = g.setdefault("metadata", self._default_metadata(g))
1985:             meta["active"] = False
1986:             g["retired"] = True
1987:             self._mark_dirty(i, payload=False, metadata=True)
1988:         meta = dict(metadata)
1989:         meta["supersedes"] = supersedes
1990:         idx = self.remember(replacement, metadata=meta,
1991:                             write_intent="user_asserted")
1992:         for i in supersedes:
1993:             self.arena.grafts[i]["metadata"]["superseded_by"] = [idx]
1994:             self._mark_dirty(i, payload=False, metadata=True)
1995:         if (mutation_plan is None
1996:                 or mutation_plan.get("apply_revision", bool(supersedes))):
1997:             self._native_apply_revision(idx, supersedes)
1998:         self._append_wal("MEMORY_CORRECT", query=query, replacement=replacement,
1999:                          supersedes=supersedes, node_id=idx)
2000:         self._mark_mutations(before)
2001:         return idx
```

## core/graft_repository.py:3273-3279
```text
3273:         if didx is None:
3274:             # fidelity abort: keep these sources unfolded (clean readers and
3275:             # routers), exempt them so the planner moves to another window.
3276:             self.folds_aborted = getattr(self, "folds_aborted", 0) + 1
3277:             for i in idxs:
3278:                 self.arena.grafts[i]["no_fold"] = True
3279:             return True
```

## core/grm_admission.py:98-118
```text
98: def is_identifier_binding(
99:     *,
100:     candidate_text: str,
101:     ordered_identifier_tokens: Sequence[str],
102:     rare_identifier_tokens: Iterable[str],
103: ) -> bool:
104:     """Frozen ADM1 identifier-binding predicate."""
105:     words = normalized_words(candidate_text)
106:     have = set(words)
107:     rare = {str(value).casefold() for value in rare_identifier_tokens}
108:     if rare:
109:         return rare <= have
110:     ordered = [str(value).casefold() for value in ordered_identifier_tokens]
111:     if not ordered:
112:         return False
113:     needle = ["current", *ordered, "value"]
114:     width = len(needle)
115:     return any(
116:         words[index:index + width] == needle
117:         for index in range(len(words) - width + 1)
118:     )
```

## core/grm_admission.py:309-342
```text
309: def policy_plan(
310:     *,
311:     ranking: Sequence[int],
312:     identified_candidates: Sequence[int],
313:     route_margin_1_2: float,
314:     margin_threshold: float = MARGIN_THRESHOLD,
315: ) -> tuple[list[int], str]:
316:     """Apply the frozen A-DEC branch precedence to one complete ranking."""
317:     ranked = [int(value) for value in ranking]
318:     identified = [int(value) for value in identified_candidates]
319:     if not ranked:
320:         return [], "empty_ranking"
321:     if not identified:
322:         return ranked[:3], "ambiguous_zero_identifier_hits_k3"
323:     if len(identified) >= 2:
324:         identified_set = set(identified)
325:         # The router may be asked for only the bounded attempt window while
326:         # identifier binding is enumerated over every eligible candidate.
327:         # Preserve router order for visible hits, then retain every identified
328:         # off-window member in the deterministic order supplied by the caller.
329:         # The branch contract is the complete identified set, not merely its
330:         # intersection with top-k.
331:         plan = [value for value in ranked if value in identified_set]
332:         plan_seen = set(plan)
333:         for value in identified:
334:             if value not in plan_seen:
335:                 plan.append(value)
336:                 plan_seen.add(value)
337:         return plan, "declared_synthesis_identified_set"
338:     if identified[0] == ranked[0]:
339:         return [ranked[0]], "exactly_one_identifier_decisive_rank1"
340:     if float(route_margin_1_2) > float(margin_threshold):
341:         return [ranked[0]], "fit_margin_decisive_rank1"
342:     return ranked[:3], "one_off_rank_identifier_insurance_k3"
```

## core/grm_admission.py:403-430
```text
403:     ordered, rare = ordered_identifier_tokens(arena, question)
404:     identified_set = {
405:         index for index in eligible
406:         if is_identifier_binding(
407:             candidate_text=str(arena.grafts[index].get("text", "") or ""),
408:             ordered_identifier_tokens=ordered,
409:             rare_identifier_tokens=rare,
410:         )
411:     }
412: 
413:     # GRM-RT1: a non-binding split-family member never outranks a binder.
414:     # Applied HERE — after the router returns and BEFORE identified / margin /
415:     # policy_plan are computed — so the demotion is visible to the A-DEC
416:     # branch decision itself, which is the whole point: on sup_solace_fresh it
417:     # turns the off-rank-1 sole binder into rank 1, and the branch from
418:     # ``one_off_rank_identifier_insurance_k3`` (plan = a competitor's chunks)
419:     # into ``exactly_one_identifier_decisive_rank1`` (plan = the fact node).
420:     #
421:     # Governed by the SAME switch P2A/P2C are, ``GRM_LSR_FIXES``.  With it OFF
422:     # this block does not run at all and the ranking stays the router's
423:     # verbatim, so the legacy path is byte-identical (test-pinned).
424:     ranking_before = list(ranking)
425:     rt1_demoted: list[int] = []
426:     rt1_members: set[int] = set()
427:     if rt1_enabled(arena) and ranking:
428:         member_probe = getattr(arena, "_split_family_members", None)
429:         if callable(member_probe):
430:             rt1_members = {int(value) for value in member_probe(ranking)}
```

## core/grm_admission.py:546-570
```text
546: def plan_priority_fit(
547:     *,
548:     plan: Sequence[int],
549:     candidates: Sequence[int],
550:     ntok: Mapping[int, int] | Any,
551:     budget: int,
552: ) -> dict[str, Any]:
553:     """Seat ``plan`` members first, in plan order, then filler.
554: 
555:     ``candidates`` is the full post-L2, post-expansion pick list for the
556:     attempt.  Members of ``plan`` that appear in it are seated FIRST in plan
557:     order; everything else is filler and consumes only the seats left over,
558:     in its own (expansion) order — the EXPANSION-ORDER truncation law of
559:     ``graft_arena.step()::fit`` is preserved for filler, which is the only
560:     population it was ever measured on (2026-06-11 score-order refutation).
561: 
562:     ``ntok`` may be a mapping or any object supporting ``ntok[index]``
563:     lookup of a per-node seat cost (the arena's ``grafts`` list satisfies
564:     this via ``grafts[i]["ntok"]`` only through the mapping adapter the
565:     callers build, so callers pass an explicit dict).
566: 
567:     Returns the complete fit receipt.  ``fit_dropped_planned`` is ``[]``
568:     unless a plan member is UNSEATABLE (its own ``ntok`` exceeds ``budget``
569:     even alone); every other unseated plan member lands in
570:     ``fit_shuttle_pending`` for the caller to serialize across trips.
```

## core/grm_admission.py:588-620
```text
588:     seated: list[int] = []
589:     used = 0
590:     pending: list[int] = []
591:     for value in seatable:
592:         n = cost(value)
593:         if used + n <= budget:
594:             seated.append(value)
595:             used += n
596:         else:
597:             # Not a drop: this member is owed its own shuttle trip.
598:             pending.append(value)
599: 
600:     seated_filler: list[int] = []
601:     dropped_filler: list[int] = []
602:     for value in filler:
603:         n = cost(value)
604:         if used + n <= budget:
605:             seated_filler.append(value)
606:             used += n
607:         else:
608:             dropped_filler.append(value)
609: 
610:     return {
611:         "fit_planned": list(plan_order),
612:         "fit_seated": sorted(seated + seated_filler),
613:         "fit_seated_planned": list(seated),
614:         "fit_seated_filler": list(seated_filler),
615:         "fit_dropped_planned": list(unseatable),
616:         "fit_dropped_filler": list(dropped_filler),
617:         "fit_shuttle_pending": list(pending),
618:         "fit_unseatable": list(unseatable),
619:         "fit_used_seats": int(used),
620:         "fit_budget": int(budget),
```

## core/graft_arena.py:692-714
```text
692:         # TOP-DOWN FILL. The plan head goes LAST in the block so its final row
693:         # is the block's final row; the rest keep their relative order below
694:         # it. The block as a whole then moves up so its last row sits at
695:         # live_shift - 1, i.e. immediately below the first live token.
696:         seat_order = picks[1:] + picks[:1]
697:         mount_pos0 = int(self.live_shift) - int(mount_ntok)
698:         head_ntok = int(self.grafts[picks[0]]["ntok"])
699:         info.update({
700:             "seat_order": [int(v) for v in seat_order],
701:             "mount_pos0": int(mount_pos0),
702:             "delta_positions": int(mount_pos0 - n_sink),
703:             # The plan head is the LAST member of the block, so it starts
704:             # head_ntok rows before the block's end.
705:             "seat_offset_plan_head": int(self.live_shift) - head_ntok,
706:             "plan_head_ntok": head_ntok,
707:             "plan_head_last_position": int(self.live_shift) - 1,
708:             "plan_head_adjacent_to_live_shift": True,
709:             "seat_rule": (
710:                 "GRM_SEAT_NEAR_LIVE: the block is filled from the TOP DOWN — "
711:                 "the plan head (picks[0]) is seated LAST so its final token "
712:                 "sits at live_shift - 1, immediately below the first live "
713:                 "token; filler and the other mounts sit below it; the sink "
714:                 "stays at [0, n_sink)"),
```

## core/graft_arena.py:1282-1292
```text
1282:     def _revision_mount_heads(self, picks):
1283:         """Return the revision heads present in one proposed mount set.
1284: 
1285:         M5's existing authoritative edge is replacement metadata
1286:         ``supersedes=[older ids]``. Walk that edge transitively: an older
1287:         candidate is removed only when one of its explicit successors is
1288:         also present. Nodes remain in route results, and a descent/request
1289:         that presents an older node without its successor still mounts it.
1290: 
1291:         Any cycle is corrupt revision metadata. Resolution fails open to the
1292:         original set rather than deleting every member of a cycle or hanging.
```

## core/graft_arena.py:1314-1339
```text
1314:         def ancestors(idx, visiting):
1315:             if idx in visiting:
1316:                 raise ValueError("cycle in explicit supersedes metadata")
1317:             if idx in memo:
1318:                 return memo[idx]
1319:             visiting.add(idx)
1320:             meta = self.grafts[idx].get("metadata") or {}
1321:             direct = self._metadata_node_ids(
1322:                 meta.get("supersedes"), len(self.grafts))
1323:             found = set(direct)
1324:             for old in direct:
1325:                 found.update(ancestors(old, visiting))
1326:             visiting.remove(idx)
1327:             memo[idx] = found
1328:             return found
1329: 
1330:         try:
1331:             for idx in unique:
1332:                 superseded_present.update(ancestors(idx, set()) & present)
1333:         except ValueError:
1334:             return unique
1335:         return [idx for idx in unique if idx not in superseded_present]
1336: 
1337:     def _resolve_revision_mounts(self, picks):
1338:         """Feature-flagged L2 mount-set resolution entry point."""
1339:         return self._revision_mount_heads(picks)
```

## core/graft_arena.py:1447-1449
```text
1447:         base = [i for i, g in enumerate(self.grafts)
1448:                 if not g.get("retired")
1449:                 and g.get("kind", "turn") != "recall"]
```

## core/graft_arena.py:3264-3293
```text
3264:     def _split_family_members(self, candidates):
3265:         """Which of ``candidates`` are width-guard split parents or children.
3266: 
3267:         Reads ONLY flags the existing split writers already set — RT1
3268:         introduces no new metadata:
3269: 
3270:         * ``metadata['width_guard_child']`` — a persisted split child
3271:           (``graft_repository._guard_deposit_width``) or an ephemeral one
3272:           (``_split_unseatable``).
3273:         * ``metadata['width_guard_parent']`` — the persisted index parent.
3274:         * ``ephemeral_split_of`` / ``ephemeral_split`` — the ephemeral pair,
3275:           for a bare arena with no repository attached.
3276: 
3277:         A node the guard never touched is not a member and the rule cannot
3278:         move it.
3279:         """
3280:         out = set()
3281:         for index in candidates:
3282:             index = int(index)
3283:             try:
3284:                 node = self.grafts[index]
3285:             except (IndexError, TypeError, KeyError):
3286:                 continue
3287:             meta = node.get("metadata") or {}
3288:             if (meta.get("width_guard_child")
3289:                     or meta.get("width_guard_parent")
3290:                     or node.get("ephemeral_split_of") is not None
3291:                     or node.get("ephemeral_split")):
3292:                 out.add(index)
3293:         return out
```

## core/graft_arena.py:4392-4414
```text
4392:         if self.caches is None:
4393:             # bootstrap: sink (+ first mounts) enter via the injection path
4394:             seat_order, seat_info = self._rs3_seat_plan(picks)
4395:             mounts = [{"h": self.sink_h}] + [
4396:                 self.grafts[i] for i in seat_order]
4397:             inj = []
4398:             # sink is host numpy; deposited grafts are device tensors
4399:             _np = lambda t: t if isinstance(t, np.ndarray) else t.numpy()
4400:             for li in range(len(self.m.layers)):
4401:                 inj.append({key: np.concatenate([_np(g["h"][li][key])
4402:                                                  for g in mounts], axis=dim)
4403:                             for key, dim in self.PAYLOAD})
4404:             # GRM-RS3 Part 2: when the lever is ON, pre-rotate the MOUNT rows
4405:             # (not the sink) by the band delta so the layer's own RoPE lands
4406:             # them adjacent to live_shift. OFF leaves `inj` exactly as the
4407:             # legacy branch built it — the same object, unrotated.
4408:             if seat_info["seat_near_live"]:
4409:                 inj = self._rs3_rotate_injection(inj, seat_info)
4410:             self._set_injection_host(inj)
4411:             self.cur_mounts = picks
4412:             self.cur_mount_n = sum(self.grafts[i]["ntok"] for i in picks)
4413:             self._commit_native_mount(picks, self.cur_mount_n)
4414:             self._rs3_last_seating = seat_info
```

## scripts/grm_det1_3_gpu.py:700-742
```text
700: def _install_lived_nodes(
701:     repo, e2e, fixture: Mapping[str, Any],
702: ) -> tuple[dict[str, int], dict[int, list[int]]]:
703:     arena = repo.arena
704:     node_to_idx = {}
705:     token_ledgers = {}
706:     for node in fixture["nodes"]:
707:         user, assistant = _split_fixture_turn(str(node["text"]))
708:         turn_text = e2e.harmony_turn(user, assistant)
709:         token_ids = [int(value) for value in arena.encode(turn_text)]
710:         before_count = len(arena.grafts)
711:         returned = arena.feed(turn_text, deposit=True)
712:         after_count = len(arena.grafts)
713:         if after_count != before_count + 1:
714:             raise DETError(
715:                 "chronological lived feed did not deposit exactly one graft: "
716:                 f"before={before_count} after={after_count}")
717:         # Non-ephemeral ArenaCache.feed deposits but intentionally returns
718:         # None; ephemeral feed returns the deposit index. Derive the common
719:         # authoritative index from the validated append.
720:         index = after_count - 1
721:         if returned is not None and int(returned) != index:
722:             raise DETError(
723:                 f"lived feed returned graft {returned}, appended graft {index}")
724:         node_to_idx[str(node["node_id"])] = index
725:         token_ledgers[index] = token_ids
726:         graft = arena.grafts[index]
727:         graft["node_id"] = index
728:         graft["kind"] = "fact"
729:         graft["metadata"] = {
730:             "kind": "fact", "active": True,
731:             "supersedes": [], "superseded_by": [],
732:         }
733:     for node in fixture["nodes"]:
734:         index = node_to_idx[str(node["node_id"])]
735:         older = [node_to_idx[str(value)] for value in node["supersedes"]]
736:         arena.grafts[index]["metadata"]["supersedes"] = older
737:         for old in older:
738:             arena.grafts[old]["metadata"]["superseded_by"].append(index)
739:     arena._bump_cuda_gqa_epoch()
740:     for index in range(len(arena.grafts)):
741:         repo._native_sync_node(index)
742:     return node_to_idx, token_ledgers
```

## scripts/grm_c7_common.py:175-191
```text
175: def score(answer, probe):
176:     text = normalize(answer)
177:     low = text.casefold().replace('\u2019', "'")
178:     abstained = bool(re.search(
179:         r"\b(?:unknown|not in memory|not (?:provided|specified)|"
180:         r"(?:do not|don't|cannot|can't) (?:know|recall|find)|"
181:         r"(?:do not|don't) have (?:that|the|this) information)\b", low))
182:     required = bool(probe['answerable'])
183:     exact = text == normalize(probe['expected'])
184:     # Disjoint errors: an answerable abstention is not ALSO an exact-value
185:     # error. Total exact-answer failure remains available for acceptance.
186:     return {'exact_correct': exact,
187:             'exact_answer_error': int(not exact),
188:             'wrong_value_error': int(required and not exact and not abstained),
189:             'abstention_error': int(required and abstained),
190:             'unsupported_answer_error': int(not required and not abstained),
191:             'abstained': abstained, 'answer': str(answer)}
```

## scripts/grm_rd1_replay.py:155-185
```text
155:     wanted = list(q['mounted_ids'])
156:     historical_ids = row['historical']['memory']['residency']['mounted_ids'] if q['side'] == 'memory' else []
157:     require(wanted == historical_ids, 'REQUEST_RESIDENCY_MISMATCH')
158:     require(len(wanted) == len(set(wanted)) and all(0 <= i < len(a.grafts) for i in wanted), 'MISSING_OR_DUPLICATE_MOUNT')
159:     require(list(a._resolve_revision_mounts(wanted)) == wanted, 'RESOLVED_MOUNTS_DIFFER')
160:     fingerprints = payload_fingerprints(repo, wanted)
161:     if baseline is not None:
162:         require(fingerprints == baseline['mounted_payload_sha256'], 'A0_PAYLOAD_MISMATCH')
163:     template = a.prompt_template
164:     a.reset_live_cache()
165:     a.eb1_begin_turn()
166:     shifts = [(layer.self_attn, hasattr(layer.self_attn, 'live_shift'),
167:                getattr(layer.self_attn, 'live_shift', None)) for layer in a.m.layers]
168:     before = len(a.grafts)
169:     try:
170:         for att, _, _ in shifts:
171:             att.live_shift = a.live_shift
172:         system = old.PREFIX.replace('Reasoning: low.', 'Reasoning: ' + q['reasoning'] + '.')
173:         def frame(user, answer):
174:             text = system + user + old.FINAL
175:             return text if answer is None else text + answer + '<|end|>'
176:         a.prompt_template = frame
177:         wrapped = a._format_step_prompt(q['prompt'])
178:         require(wrapped == q['wrapped_prompt'], 'WRAPPED_PROMPT_MISMATCH')
179:         require(list(a.stop_sequences) == q['stops'], 'STOP_SEQUENCE_MISMATCH')
180:         a._ensure_h(wanted)
181:         require(all(a.grafts[i].get('h') is not None for i in wanted), 'MOUNT_PAYLOAD_NOT_LOADED')
182:         token_ids = list(a.encode(wrapped))
183:         answer, info = a._attempt(q['prompt'], wanted, q['answer_budget'], False,
184:                                   tuple(q['stops']), defer_memory=True)
185:         require(list(a.cur_mounts) == wanted, 'ACTUAL_MOUNTS_DIFFER')
```

