#!/usr/bin/env bash
# Replicate the frozen evaluation samples to another machine, and VERIFY it.
#
# `evaluator/runs/` is git-ignored, ~48 MB, and irreplaceable: it is every
# (model, task) completion ever paid for -- hundreds of dollars and tens of
# hours of sampling. Re-creating it means re-buying it. It has existed as a
# single copy on a single disk, and on 2026-08-06 that copy was recovered from
# a powered-off VM by luck rather than by plan.
#
# The verification is the point, not a nicety. A backup nobody has checked is a
# belief, not a backup: in July a SQLite ledger was copied while its service was
# running and the copy silently lost a third of its rows -- discovered only when
# the numbers were needed. So this script writes a manifest of per-file sha256
# BEFORE sending, and `verify` re-hashes the remote copy against it.
#
# Usage:
#   scripts/backup_runs.sh push   <ssh-host> [remote-dir]
#   scripts/backup_runs.sh verify <ssh-host> [remote-dir]
#
#   scripts/backup_runs.sh push   eval103
#   scripts/backup_runs.sh verify eval103
#
# Deliberately NOT included: runs/secrets/. Credentials do not belong in a
# convenience copy on another machine -- they are rotated, not restored.

set -euo pipefail

CMD="${1:-}"
HOST="${2:-}"
REMOTE="${3:-~/fusion-gateway-backup}"
SRC="evaluator/runs"
MANIFEST=".backup-manifest.sha256"

[ -n "${CMD}" ] && [ -n "${HOST}" ] || { sed -n '2,26p' "$0" | sed 's/^# \?//'; exit 2; }
[ -d "${SRC}" ] || { echo "no ${SRC} here — run from the repo root" >&2; exit 1; }

manifest() {
  # Sorted, relative paths: the same tree must hash identically on both ends
  # regardless of directory read order or where it is rooted.
  ( cd "${SRC}" && find . -type f ! -name "${MANIFEST}" -print0 \
      | sort -z | xargs -0 sha256sum )
}

case "${CMD}" in
  push)
    echo "→ hashing $(find "${SRC}" -type f | wc -l) files in ${SRC}"
    manifest > "${SRC}/${MANIFEST}"
    echo "→ replicating to ${HOST}:${REMOTE}"
    ssh "${HOST}" "mkdir -p ${REMOTE}"
    # -c: checksum, not mtime. A copy that differs in content but matches in
    # size and timestamp is exactly the kind of silent corruption this exists
    # to catch, and mtime-based rsync would skip it.
    rsync -a -c --delete "${SRC}/" "${HOST}:${REMOTE}/"
    echo "→ verifying the copy that just landed"
    exec "$0" verify "${HOST}" "${REMOTE}"
    ;;

  verify)
    echo "→ re-hashing ${HOST}:${REMOTE} against its manifest"
    if ! ssh "${HOST}" "cd ${REMOTE} && test -f ${MANIFEST}"; then
      echo "✗ no manifest at ${HOST}:${REMOTE} — nothing was ever pushed there" >&2
      exit 1
    fi
    # Runs on the REMOTE, over the remote's own bytes. Hashing locally and
    # comparing manifests would only prove the manifest travelled intact.
    if ssh "${HOST}" "cd ${REMOTE} && sha256sum --quiet -c ${MANIFEST}"; then
      n=$(ssh "${HOST}" "wc -l < ${REMOTE}/${MANIFEST}")
      sz=$(ssh "${HOST}" "du -sh ${REMOTE} | cut -f1")
      echo "✓ ${n} files, ${sz}, every hash matches"
    else
      echo "✗ the remote copy does not match its manifest. Do not trust it." >&2
      exit 1
    fi
    ;;

  *) echo "usage: $0 {push|verify} <ssh-host> [remote-dir]" >&2; exit 2 ;;
esac
