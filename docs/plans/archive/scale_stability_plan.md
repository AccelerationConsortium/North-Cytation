# Scale stability — investigation plan

Date: 2026-11-18 · Status: **active, nothing started.**

## Problem

Scale volatility is hurting data collection. The likely cause is the clamp gripping and
ungripping the vial when it's moved to the clamp. That step has to stay, because it stops
the vial moving, but it makes the vial wobble.

We never used to need a long wait for the scale to settle.

## Goal

1. Find out why the scale is stable sometimes and not others, including why the 15 s
   stabilization check passes in some runs and fails in others.
2. Find a fix for the physical cause, not a longer wait.

## Steps

1. **Watch the vial in the clamp.** Repeat the move to the clamp several times while the
   scale is logged. Note how the vial wobbles and how it ends up sitting, for runs where
   the scale settles and runs where it doesn't.
2. **Compare stable and unstable cases.** Look for what differs between them.
3. **Try physical changes** to how the vial is gripped and released in the clamp, and
   check each one against the scale log from step 1.
4. **Keep the change that gives a stable scale** without needing a long wait.
