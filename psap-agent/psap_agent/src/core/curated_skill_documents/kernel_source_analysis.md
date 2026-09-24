# Kernel and source analysis

Use this workflow to map changed kernels to vLLM code, inspect source diffs,
verify dependencies, answer questions about vLLM releases, or inspect a vLLM PR.

## Evidence sources

- Use `get_kernel_call_stacks` when a profile was collected with stacks. These
  call paths are the strongest source attribution.
- Otherwise use `map_kernel_to_vllm_code` after profiler analysis. Treat its
  heuristic mappings as likely locations, not proof.
- Use `get_kernel_categories` when category terminology needs interpretation.
- Use `correlate_kernel_with_changes` after a profile identifies a changed
  kernel to guide the code investigation; verify its suggestions with source
  tools before making a causal claim.
- Use `fetch_vllm_source` to read an exact source file or dependency pin at a
  specific version, and `get_vllm_code_diff` to inspect the actual diff for a
  relevant file.
- Use `get_vllm_release_notes` for release-content questions, and
  `compare_vllm_versions` for a change summary across releases. Do not answer
  version-specific release content from general model knowledge.
- Use `get_vllm_pull_request` when a user asks about a specific PR or a release
  note refers to one. Request patches when the actual code change matters.
- Use `get_version_mappings` when RHAIIS-to-vLLM mapping is needed. If a source
  or diff tool cannot resolve an RHAIIS version, retrieve the mapping and retry
  with the vLLM tag.

## Verify dependency chains

For a dependency claim, retrieve every link in the chain. For example, fetch
vLLM's `requirements/cuda.txt` or `pyproject.toml` for its PyTorch pin, then
retrieve the upstream PyTorch pin file using `fetch_vllm_source` with
`repo='pytorch/pytorch'`. If a source cannot be retrieved, say that the chain
is incomplete rather than filling the gap with training knowledge.

## Attribute with care

Use source changes and release notes to explain a profiler observation, not to
replace profiler evidence. Cite the exact tool-retrieved file, diff, or PR and
state when the connection remains a hypothesis.

For a deep root-cause analysis, source investigation follows profiler and log
evidence. Fetch source for both versions and inspect the diff for each claimed
mechanism. If a source tool returns an error, report that limitation rather than
claiming the source was examined.
