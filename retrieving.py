from transformers.utils.hub import cached_file

index_path = cached_file(
    "facebook/rag-token-nq",
    filename="index",
    revision='commit/c269b105d2322e9386b629a0a8663d20863a5167',
    cache_dir="./",
    force_download=True,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
)
