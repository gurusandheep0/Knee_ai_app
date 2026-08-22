# Model integration point

The portable demonstrator intentionally does not ship an unvalidated medical AI
model. Bundled synthetic cases already include deterministic masks so the whole
measurement and reporting workflow can be demonstrated offline.

For real image inference, add a validated ONNX model here and implement its exact
preprocessing, output-label mapping, quality checks, and transformation back to
the original physical coordinate system. Do not assume that an arbitrary knee
model accepts every MRI or CT protocol.

Expected application mask names are `femur`, `tibia`, and `meniscus`, aligned with
the input volume and stored using the `(ML, SI, AP)` project convention.
