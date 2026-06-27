# Codex Notes

- Do not run the DeepSeek OCR service or model inference in this workspace.
- This machine is CPU-only for our purposes, while the project expects an NVIDIA CUDA GPU.
- Local validation should be limited to static inspection or lightweight non-GPU checks unless the user explicitly provides a GPU runtime.
- Prompt under test:

```text
<image>
<|grounding|>Extract all visible document content into clean Markdown. Preserve reading order. Transcribe normal text exactly. If the page contains mathematics, physics, chemistry, or technical notation, write every formula in valid LaTeX using inline `$...$` or display `$$...$$` form as appropriate. Preserve equation numbers, variables, units, superscripts, subscripts, fractions, roots, vectors, matrices, and symbols. For tables, return Markdown tables. For any photo, figure, diagram, chart, graph, or boxed visual region, include an image/figure item with a short description and its bounding box points as `[x1,y1,x2,y2]` in image pixel coordinates. If a boxed question, boxed answer, highlighted region, or callout exists, return its box points too. Do not invent missing text. Mark unreadable parts as `[illegible]`.
```
