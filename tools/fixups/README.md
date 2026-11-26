# Repair-pass scripts

One directory per recorded repair pass: `tools/fixups/<run_id>/` with the pass's scripts and a `NOTES.md`
(run id `YYYYMMDD-<model>-<topic>`). The attempt store references these scripts by path and git blob sha, so a
script that produced imported attempts is never edited again; copy it to `<name>_v2.py` instead.
The workflow is in CLAUDE.md (also read as AGENTS.md / GEMINI.md) under "Repair passes".
