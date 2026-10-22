The two CSVs at this directory root are the pair immediately BEFORE this second cleanup.
staged/ holds the cleaned pair. row_manifest.csv maps original, previous and current row positions.
The original unfiltered archive remains intact at the prior_archive path in manifest.json.
Rollback: pause master writers, preserve any subsequent recordings, verify archive hashes, then restore BOTH root CSVs.
If activation was interrupted, verify/restore the pair before resuming writes. No metadata repairs were performed.
