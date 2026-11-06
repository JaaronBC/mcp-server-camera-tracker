# Compact Runtime Skill — Independent Track Verification

## Objective

Independently verify whether evidence supports, contradicts, or is inconclusive for a referenced tracking identity claim, without mutating tracking state.

## Verification Boundary

- Observation verifies claims; Tracking decides associations.
- Never emit imperative correction commands; provide evidence-grounded verdicts and rationale.

## Evidence Inputs

- Use only referenced tracking evidence, bounded history, image evidence (if available), provenance metadata, and source evidence references.
- Do not infer unavailable measurements or fabricate frame observations.

## Two-Layer Reasoning (Keep Separate)

1. **Structured consistency layer**:
   - Check motion/position/depth coherence against the claimed identity.
   - This supports plausibility, not identity proof.
2. **Visual identity layer**:
   - Assess whether image content actually supports/contradicts identity continuity.
   - Account for occlusion, crossings, similarity, low light, blur, and compression.

## Adjudication Policy

- **supported**: claim is aligned and alternatives are less plausible.
- **contradicted**: materially conflicting evidence with explicit contradictory references.
- **inconclusive**: evidence is insufficient, ambiguous, or mixed.
- Be conservative: when identity evidence is weak, choose inconclusive.

Strong structured contradictions may be reported even when visual identity confirmation is unavailable, but do not claim visual confirmation in that case.

## Competing Identity Handling

- Evaluate plausible alternatives explicitly.
- Increase uncertainty or contradiction pressure when alternatives remain similarly supported.
- Flag possible ID switch when crossing/occlusion/trajectory behavior suggests it.

## `visual_content_inspected` Definition

- `true` only when image content was actually inspected, regardless of image quality.
- `false` when no image content was inspected.
- A frame reference alone is not proof of inspection.

## Visual Availability Rule

- If visual evidence is missing/unusable, do structured-only assessment and avoid visual identity confirmation.
- Keep conclusions conservative under reduced visual reliability.

## Uncertainty, Limitations, and Provenance

- Uncertainty captures unresolved identity ambiguity.
- Limitations capture concrete evidence constraints (occlusion, low quality, missing visibility, insufficient resolution/time window).
- Every material support/contradiction claim should map to available evidence references.
- If provenance is weak or missing, downgrade confidence and add limitations.

## Collaboration Output Intent

- Provide clear support/contradiction rationale so Tracking can reconsider associations.
- Preserve independence and neutrality when evidence is mixed.
