Originals are the two CSVs at this directory root. staged/ contains the cleaned pair.
row_manifest.csv maps every original occurrence to its classification and retained row.
classification_snapshot.csv preserves full evidence, including original record JSON.
To roll back: pause master writers, verify original hashes in manifest.json, archive the
then-current active pair first (new recordings may exist), and restore BOTH original CSVs.
If manifest status is activation_in_progress, verify both active hashes before resuming.
No stage/overaspirate edits or analyzer code changes are included. Unresolved rows remain.
