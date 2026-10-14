# Local dattri patch

`maestro-generated-music.patch` preserves the existing local change in dattri's MAESTRO loader: download generated music only when `generated_music=True`.

It applies to the pinned submodule revision ef50a35e6cdfa358ca104b7655ef15dcb9650afe. The patch is stored in our repository; it is not an upstream dattri commit.

From the whiteboard repository root, on a fresh checkout:

```powershell
git submodule update --init --recursive
git -C vendor/dattri apply --check ../../patches/maestro-generated-music.patch
git -C vendor/dattri apply ../../patches/maestro-generated-music.patch
```

If the change is already present, this check succeeds instead:

```powershell
git -C vendor/dattri apply --reverse --check ../../patches/maestro-generated-music.patch
```

The current working checkout already has the patch applied. Do not apply it a second time. A modified submodule status is expected after applying it. Preserve the local change; do not push it to the upstream repository.
