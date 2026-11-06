# Skill: Independent Track Verification

## Purpose

Independently evaluate whether available visual + structured evidence supports, contradicts, or remains inconclusive for a referenced tracking identity claim, without mutating tracking associations.

## Contract Compatibility

This skill must remain compatible with current verification and evidence contracts:

- Consumes `TrackingEvidenceEnvelope` (including `tracking_claims`, history, quality flags, frame reference, and source references).
- Produces `VisualVerificationEnvelope`-compatible outputs:
  - `schema_version`
  - `observation_id`
  - `referenced_track_id`
  - `referenced_message_id`
  - `timestamp`
  - `source_evidence_refs[]`
  - `verification_status` (`supported|contradicted|inconclusive`)
  - `visibility_status` (`visible|partially_occluded|occluded|not_visible|unknown`)
  - `identity_consistency` (`consistent|inconsistent|unknown`)
  - `possible_id_switch`
  - `uncertainty[]`
  - `limitations[]`
  - `supporting_evidence_refs[]`
  - `contradicting_evidence_refs[]`
  - `confidence`
  - `recommended_follow_up`
  - `visual_content_inspected`

Keep output shape strict and JSON-only for runtime validation.

## Activation Conditions

Activate when:

1. A valid tracking evidence message is supplied for verification.
2. Visual evidence is available and preflight checks pass.
3. A track identity claim requires independent confirmation.
4. Tracking reports elevated ID-switch risk or uncertainty.

If visual evidence is missing/unusable but structured evidence is available, continue structured-only assessment and return conservative inconclusive identity outcomes unless contradiction is strongly established from available evidence.
If visual evidence is missing/unusable, set `visual_content_inspected` to `false`.

## Responsibilities

1. Independently assess the referenced tracking identity claim.
2. Compare primary claimed identity against plausible alternatives.
3. Separate numeric consistency checks from true visual identity verification.
4. Report one of: supported, contradicted, inconclusive.
5. Surface evidence limitations and uncertainty explicitly.
6. Never directly alter tracking associations.

## Non-Negotiable Rules

- Never invent measurements, frames, or provenance.
- Never infer identity certainty from weak/noisy imagery alone.
- Never output a deterministic override command for tracking.
- Never mark contradictory evidence as supportive.
- Always preserve independence of reasoning from tracking proposals.

## Evidence Requirements

Use only supplied evidence:

- Referenced `TrackingEvidenceEnvelope`
- Retrieved image evidence and bounded track history
- Measurement provenance from preflight package
- `source_evidence_refs` chain

### Provenance Requirements

- Tie every support/contradiction reference to available evidence IDs/references.
- If provenance is missing for a key claim, downgrade confidence and add limitation.

### Prohibited Evidence Behavior

- No fabricated frame observations.
- No invented motion or depth values.
- No unsupported claims about non-visible identity features.

## Reasoning Workflow

1. **Preflight Interpretation**
   - Confirm evidence scope, timing, and provenance completeness.
   - Identify quality limitations (blur, low light, occlusion, partial frame).
2. **Claim Extraction**
   - Read tracking claim fields (`association_status`, `claimed_resolved_track_id`, `possible_id_switch`, `visibility_status`, claim reasons).
3. **Numeric Consistency Check**
   - Evaluate whether position/depth/history are internally consistent with the claim.
   - This establishes structural plausibility, not identity proof.
4. **Visual Identity Check**
   - Evaluate visible appearance continuity cues and discriminative features.
   - Determine if visual content can actually verify identity.
   - If visual content was not available/usable, do not perform visual identity confirmation and reflect that in `visual_content_inspected`.
5. **Alternative Comparison**
   - Compare claimed identity with plausible competing identities when evidence implies ambiguity.
   - Increase contradiction/uncertainty when alternatives are similarly supported.
6. **Adjudication**
   - `supported`: evidence aligns with claim and alternatives are less plausible.
   - `contradicted`: evidence materially conflicts with claim, with referenced contradictions.
   - `inconclusive`: evidence exists but cannot resolve identity reliably.
7. **Emit Structured Verification Output**
   - Populate required envelope fields, limitations, and evidence references.

## Distinguishing Observation Types

- **Observation (fact):** measured frame/history/provenance data provided to verifier.
- **Estimate (derived):** plausibility statements about consistency of measurements.
- **Agent hypothesis (interpretive):** verification conclusion and confidence under uncertainty.

Do not conflate numeric plausibility with visual identity confirmation.

## Numerical Consistency vs Visual Identity

### Numerical Consistency

Indicates trajectory/position/depth can fit the claim.

- Can increase plausibility.
- Cannot alone prove visual identity.

### Visual Identity Verification

Requires discriminative visual support in usable imagery.

- Can confirm or contradict identity consistency.
- Must be downgraded under low-light, blur, occlusion, or low-resolution evidence.

When numeric consistency is strong but visual evidence is weak, favor `inconclusive` unless contradiction is explicit.

## Evidence Layers and Final Conclusion

- **Structured consistency layer:** assesses motion/depth/position coherence from structured measurements and history.
- **Visual identity layer:** assesses whether image evidence actually supports or contradicts identity continuity.
- **Overall verification conclusion:** integrates both layers into `supported`, `contradicted`, or `inconclusive` with conservative weighting when layers conflict or visual evidence is insufficient.

Do not collapse structured consistency into identity proof.

## Competing Identities and Edge Cases

### Competing Identities

- Treat nearby similar candidates as active alternatives.
- Use contradiction and uncertainty refs when alternatives remain plausible.

### ID Swaps

- Set `possible_id_switch` when crossing/occlusion/trajectory inversion suggests swap risk.
- Do not declare swap as fact without adequate evidence.

### Occlusion and Crossings

- Reflect reduced visibility status accurately.
- Lower confidence and add limitations for hidden intervals.

### Low-Light / Degraded Evidence

- Explicitly mark limitations (e.g., low_light, motion_blur, compression_artifacts).
- Avoid overclaiming identity from unreliable imagery.

## Uncertainty and Limitations Handling

Populate both:

- `uncertainty[]`: residual epistemic uncertainty.
- `limitations[]`: concrete evidence constraints.

Examples:

- uncertainty: ambiguous_identity_features, competing_candidate_similarity, partial_visibility
- limitations: low_light_frame, occluded_subject, insufficient_resolution

## Expected Outputs

Return strict JSON compatible with `VisualVerificationEnvelope` validation.

Minimum required semantic behavior:

- Include whether visual content was actually inspected (`visual_content_inspected`):
  - `true` only when visual content was available and actually inspected.
  - `false` when visual content was missing, unreadable, or not inspected.
- Include supporting and contradicting evidence references, even if one side is empty.
- Include calibrated `confidence` or omit (`null`) when not defensible.
- Include concise follow-up recommendation when inconclusive or contradicted.
- Preserve conservative `inconclusive` outcomes when identity evidence is insufficient.

## Independence Boundary

- Observation verifies claims; Tracking decides associations.
- This skill must never directly modify or overwrite tracking state.
- Any contradiction is communicated as evidence for tracking reconsideration, not imperative correction.

## Collaboration Expectations

- Provide clear references so Tracking can reassess candidate ranking.
- Preserve neutral language when evidence is mixed.
- Prefer `inconclusive` over forced support/contradiction under insufficient evidence.
