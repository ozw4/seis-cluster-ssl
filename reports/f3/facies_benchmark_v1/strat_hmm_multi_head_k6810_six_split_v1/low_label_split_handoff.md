# M4 six-split handoff

## Formal result

Formal result: `M4_MH_SPLIT_HOLD`
- HOLD is preserved as the formal six-split confirmatory result.

## Project decision

Project decision: `ADOPT_MH_NOCONS_FOR_M5`
- Adoption is a project decision, separate from the formal confirmatory status.

## Selected baseline

- `mh_nocons`
- `strat_hmm_pretext_mh_k6810_nocons_topblock1_distill_v1`

## Reference models

- current K6 (`m1_current_k6`)
- MAE (`mae`)

## Next milestone

- `M5_U_SOFT_POSTERIOR` - M5-U soft posterior.
- Posterior-aware soft multi-resolution HMM pretraining is planned; its effectiveness is unverified.

## Carry forward

- mh_nocons
- current K6
- MAE
- Existing original-split and six-split downstream artifacts

## Do not carry forward as primary candidate

- mh_cons010

## No longer required as a gate

- Decoder seeds `42001/42002`; they remain optional diagnostics.
