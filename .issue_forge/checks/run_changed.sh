#!/usr/bin/env bash
set -euo pipefail

readonly WORK_EXCLUDE_PATHSPEC=':(exclude).work'
readonly VENDOR_EXCLUDE_PATHSPEC=':(exclude)vendor/issue_forge'
readonly PORTABLE_MARKERS='not slow and not requires_segy and not requires_cuda'

fail() {
  printf '%s\n' "$1" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "Missing required check command: $1"
}

collect_changed_files() {
  local base_ref="$1"

  {
    git diff --name-only "$base_ref" -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
    git diff --name-only --cached -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
    git diff --name-only -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
    git ls-files --others --exclude-standard -- . "$WORK_EXCLUDE_PATHSPEC" "$VENDOR_EXCLUDE_PATHSPEC"
  } | awk 'NF && !seen[$0]++'
}

run_shellcheck_if_needed() {
  local -a shell_targets=("$@")

  if [[ "${#shell_targets[@]}" -eq 0 ]]; then
    printf 'shellcheck: skipped (no shell targets changed)\n'
    return 0
  fi

  require_command shellcheck
  printf 'shellcheck: %s target(s)\n' "${#shell_targets[@]}"
  shellcheck -x "${shell_targets[@]}"
}

run_pytest_if_needed() {
  local scope="$1"
  shift
  local -a test_targets=("$@")
  local -a pytest_args=(-q)
  local -A seen_targets=()
  local target

  if [[ "$scope" == none && "${#test_targets[@]}" -eq 0 ]]; then
    printf 'pytest: skipped (no Python-related changes)\n'
    return 0
  fi

  require_command pytest
  if [[ "$scope" == portable ]]; then
    pytest_args+=(-m "$PORTABLE_MARKERS")
  fi
  if [[ "$scope" == none ]]; then
    for target in "${test_targets[@]}"; do
      if [[ -z "${seen_targets[$target]:-}" ]]; then
        pytest_args+=("$target")
        seen_targets["$target"]=1
      fi
    done
  fi
  printf 'pytest:'
  printf ' %q' pytest "${pytest_args[@]}"
  printf '\n'
  pytest "${pytest_args[@]}"
}

main() {
  local base_ref
  local path
  local pytest_scope=none
  local -a changed_files=()
  local -a shell_targets=()
  local -a test_targets=()

  if [[ "$#" -ne 1 ]]; then
    fail "Usage: $0 <base-ref>"
  fi

  base_ref="$1"

  git rev-parse --verify "$base_ref" >/dev/null 2>&1 || fail "Missing base ref for checks: $base_ref"

  mapfile -t changed_files < <(collect_changed_files "$base_ref")

  if [[ "${#changed_files[@]}" -eq 0 ]]; then
    printf 'No changes detected relative to %s\n' "$base_ref"
    return 0
  fi

  printf 'Changed files relative to %s:\n' "$base_ref"
  printf ' - %s\n' "${changed_files[@]}"

  for path in "${changed_files[@]}"; do
    case "$path" in
      *.sh)
        [[ -f "$path" ]] && shell_targets+=("$path")
        ;;
    esac

    case "$path" in
      pytest.ini|pyproject.toml|setup.py|setup.cfg|tox.ini|requirements*.txt|Pipfile|Pipfile.lock|poetry.lock|uv.lock)
        pytest_scope=full
        ;;
      tests/test_*.py|tests/*/test_*.py)
        if [[ -f "$path" ]]; then
          test_targets+=("$path")
        elif [[ "$pytest_scope" != full ]]; then
          pytest_scope=portable
        fi
        ;;
      *.py|tests/*)
        [[ "$pytest_scope" == full ]] || pytest_scope=portable
        ;;
      .issue_forge/checks/run_changed.sh)
        test_targets+=(tests/seis_ssl_cluster/test_issue_forge_checks.py)
        ;;
    esac
  done

  run_shellcheck_if_needed "${shell_targets[@]}"
  run_pytest_if_needed "$pytest_scope" "${test_targets[@]}"
}

main "$@"
