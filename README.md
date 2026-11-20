##Hallucination Behavior in Multimodal LLMs Across Agricultural Image Interpretation and Generation Tasks

This repository contains the code and data used to produce the results reported in the accompanying manuscript.

##Purpose

The code supports a pilot study of how multimodal large language models (MLLMs) hallucinate on agricultural imagery. It is used to:

Evaluate image interpretation. Run four open-source MLLMs (Gemma-2B, LLaVA-1.5, Qwen-2.5-VL, MiniCPM-V) on expert-labeled tomato leaf images (three-class disease classification) and strawberry images (fruit presence, ripeness, and counting), in zero-shot and 1-shot settings.
Evaluate image generation. Test two commercial generative models (GPT-5 and Gemini 2.5 Flash) with biologically impossible prompts under four framing conditions (T1–T4).
Classify errors. Assign incorrect outputs to the Agricultural Hallucination Taxonomy (AHT): Object-Level, Attribute-Level, Semantic-Relational, and Physiological-Contradiction.
Score generated images. Compute the five-criterion Biological Fidelity Score (BFS) and generation rates.
