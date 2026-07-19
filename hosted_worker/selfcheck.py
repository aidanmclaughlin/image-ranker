from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import distributions

from image_ranker.ml import ENCODER, MODEL_NAME, PRETRAINED, _OpenClipRuntime

from .encoder import MANIFEST_PATH, RENDITION_SCHEMA


def _model_state_digest(runtime: _OpenClipRuntime) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(runtime.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(value.view(runtime.torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _package_inventory() -> dict[str, str]:
    return dict(
        sorted(
            (
                distribution.metadata["Name"].casefold(),
                distribution.version,
            )
            for distribution in distributions()
            if distribution.metadata.get("Name")
        )
    )


def main() -> int:
    runtime = _OpenClipRuntime(device="cpu")
    if runtime.device != "cpu":
        raise RuntimeError("snapshot OpenCLIP self-check did not select CPU")
    manifest = {
        "base_encoder": ENCODER,
        "model_name": MODEL_NAME,
        "pretrained": PRETRAINED,
        "rendition_schema": RENDITION_SCHEMA,
        "model_state_sha256": _model_state_digest(runtime),
        "preprocess": repr(runtime.preprocess),
        "python": platform.python_version(),
        "packages": _package_inventory(),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"encoder": ENCODER, "fingerprint": manifest["fingerprint"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
